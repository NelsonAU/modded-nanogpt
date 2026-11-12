import os
from functools import lru_cache
import torch
import triton
import triton.language as tl

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:
    TensorDescriptor = None

@lru_cache(maxsize=1)
def has_tma_support() -> bool:
    if not torch.cuda.is_available() or TensorDescriptor is None:
        return False
    try:
        return torch.cuda.get_device_capability()[0] >= 9
    except RuntimeError:
        return False

# Note: While compute capability 8.9 (Ada Lovelace) cards do support FP8, we
# also need column-major FP8 _scaled_mm and the ce_fwd_bwd kernel, which is
# only available on >= 9.0 (Hopper).
@lru_cache(maxsize=1)
def use_fp8() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability()[0] >= 9
    except RuntimeError:
        return False

# -----------------------------------------------------------------------------
# Triton kernel for MLP: relu(x @ W1.T)^2, by @andrewbriand, @jrauvola

@triton.jit
def linear_relu_square_kernel(a_desc, b_desc, c_desc, aux_desc,
                                 M, N, K,
                                 BLOCK_SIZE_M: tl.constexpr,
                                 BLOCK_SIZE_N: tl.constexpr,
                                 BLOCK_SIZE_K: tl.constexpr,
                                 GROUP_SIZE_M: tl.constexpr,
                                 NUM_SMS: tl.constexpr,
                                 FORWARD: tl.constexpr,
                                 ):
    dtype = tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)

        c0 = acc0.to(dtype)
        if not FORWARD:
            c0_pre = aux_desc.load([offs_am_c, offs_bn_c])
            c0 = 2 * c0 * tl.where(c0_pre > 0, c0_pre, 0)

        c_desc.store([offs_am_c, offs_bn_c], c0)

        if FORWARD:
            c0_post = tl.maximum(c0, 0)
            c0_post = c0_post * c0_post
            aux_desc.store([offs_am_c, offs_bn_c], c0_post)

        c1 = acc1.to(dtype)
        if not FORWARD:
            c1_pre = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
            c1 = 2 * c1 * tl.where(c1_pre > 0, c1_pre, 0)

        c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)

        if FORWARD:
            c1_post = tl.maximum(c1, 0)
            c1_post = c1_post * c1_post
            aux_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1_post)


def linear_relu_square(a, b, aux=None):
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    FORWARD = False
    if aux is None:
        FORWARD = True
        aux = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    num_stages = 4 if FORWARD else 3
    num_warps = 8

    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = TensorDescriptor.from_tensor(c, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    aux_desc = TensorDescriptor.from_tensor(aux, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])

    def grid(META):
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
        ), )

    linear_relu_square_kernel[grid](
        a_desc, b_desc, c_desc, aux_desc,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=1,
        NUM_SMS=NUM_SMS,
        FORWARD=FORWARD,
        num_stages=num_stages,
        num_warps=num_warps
    )

    if FORWARD:
        return c, aux
    else:
        return c

class FusedLinearReLUSquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W1, W2):
        pre, post = linear_relu_square(x.view((-1, x.shape[-1])), W1)
        x3 = post @ W2
        ctx.save_for_backward(x, W1, W2, pre, post)
        return x3.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, pre, post = ctx.saved_tensors
        dW2 = post.T @ grad_output
        dpre = linear_relu_square(grad_output.view((-1, grad_output.shape[-1])), W2, aux=pre)
        dW1 = dpre.T @ x
        dx = dpre @ W1
        return dx.view(x.shape), dW1, dW2


# -----------------------------------------------------------------------------
# Tiled transpose copy kernel: dst (N, M) = src (M, N).T
# Uses coalesced reads from src and coalesced writes to dst via tl.trans().
# Replaces PyTorch's elementwise copy_ which uses a naive 75k-block kernel
# with non-coalesced writes, saturating all SMs and blocking NCCL.

@triton.jit
def _transpose_copy_kernel(
    src_ptr, dst_ptr,
    M, N,
    src_stride_m, src_stride_n,
    dst_stride_0, dst_stride_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Coalesced read from src (M, N)
    tile = tl.load(
        src_ptr + offs_m[:, None] * src_stride_m + offs_n[None, :] * src_stride_n,
        mask=mask, other=0.0,
    )

    # Coalesced write to dst (N, M): dst[n, m] = src[m, n]
    mask_T = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(
        dst_ptr + offs_n[:, None] * dst_stride_0 + offs_m[None, :] * dst_stride_1,
        tl.trans(tile), mask=mask_T,
    )


def transpose_copy(src: torch.Tensor, dst: torch.Tensor):
    """Tiled transpose copy: dst = src.T where src is (M, N) and dst is (N, M).

    Uses a 64x128 tiled Triton kernel with coalesced reads AND writes,
    achieving near memory-bandwidth-limited performance.
    """
    assert src.ndim == 2 and dst.ndim == 2
    M, N = src.shape
    assert dst.shape == (N, M), f"Expected dst shape ({N}, {M}), got {dst.shape}"

    BLOCK_M, BLOCK_N = 64, 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _transpose_copy_kernel[grid](
        src, dst,
        M, N,
        src.stride(0), src.stride(1),
        dst.stride(0), dst.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=2,
    )


# -----------------------------------------------------------------------------
# Tiled transpose-add kernel: dst (M, N) += src (N, M).T
# Same tiling strategy as transpose_copy but with a fused read-add-write.
# Replaces PyTorch's .add_(src.T) which uses the same 75k-block elementwise
# kernel with non-coalesced reads from the transposed operand.

@triton.jit
def _transpose_add_kernel(
    src_ptr, dst_ptr,
    M, N,
    src_stride_m, src_stride_n,
    dst_stride_0, dst_stride_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Coalesced read from src (M, N)
    src_tile = tl.load(
        src_ptr + offs_m[:, None] * src_stride_m + offs_n[None, :] * src_stride_n,
        mask=mask, other=0.0,
    )

    # Coalesced read-add-write on dst (N, M): dst[n, m] += src[m, n]
    mask_T = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    dst_ptrs = dst_ptr + offs_n[:, None] * dst_stride_0 + offs_m[None, :] * dst_stride_1
    dst_tile = tl.load(dst_ptrs, mask=mask_T, other=0.0)
    tl.store(dst_ptrs, dst_tile + tl.trans(src_tile), mask=mask_T)


def transpose_add(src: torch.Tensor, dst: torch.Tensor):
    """Tiled transpose-add: dst += src.T where src is (M, N) and dst is (N, M).

    Uses a 32x32 tiled Triton kernel with coalesced access on both src and dst,
    replacing PyTorch's .add_(src.T) which has non-coalesced reads from the
    transposed operand.
    """
    assert src.ndim == 2 and dst.ndim == 2
    M, N = src.shape
    assert dst.shape == (N, M), f"Expected dst shape ({N}, {M}), got {dst.shape}"

    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _transpose_add_kernel[grid](
        src, dst,
        M, N,
        src.stride(0), src.stride(1),
        dst.stride(0), dst.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )


@triton.jit
def fused_softcapped_entropy_fwd_kernel(
    logits_ptr, losses_ptr, lse_ptr, targets_ptr, mtp_weights_ptr,
    stride_logits_n, stride_logits_v,
    n_rows, n_cols, n_predict,
    A, B, C,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0).to(tl.int64)
    logits_row_ptr = logits_ptr + row_idx * stride_logits_n

    neg_inf = tl.cast(-float('inf'), tl.float32)
    max_val = neg_inf
    sum_exp = tl.cast(0.0, tl.float32)

    A_f32 = tl.cast(A, tl.float32)
    B_f32 = tl.cast(B, tl.float32)
    C_f32 = tl.cast(C, tl.float32)
    inv_C = 1.0 / C_f32
    B_div_C = B_f32 * inv_C

    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        val = tl.load(logits_row_ptr + cols, mask=mask, other=neg_inf).to(tl.float32)
        z = A_f32 * tl.sigmoid(val * inv_C + B_div_C)
        z = tl.where(mask, z, neg_inf)
        curr_max = tl.max(z, axis=0)
        new_max = tl.maximum(max_val, curr_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(tl.exp(z - new_max), axis=0)
        max_val = new_max

    lse = max_val + tl.log(sum_exp)
    tl.store(lse_ptr + row_idx, lse)

    total_loss = tl.cast(0.0, tl.float32)
    for k in range(n_predict):
        target_idx = row_idx + k
        if target_idx < n_rows:
            weight = tl.load(mtp_weights_ptr + k).to(tl.float32)
            if weight > 0:
                target = tl.load(targets_ptr + target_idx).to(tl.int32)
                if target >= 0 and target < n_cols:
                    val_target = tl.load(logits_row_ptr + target).to(tl.float32)
                    z_target = A_f32 * tl.sigmoid(val_target * inv_C + B_div_C)
                    total_loss += weight * (lse - z_target)

    tl.store(losses_ptr + row_idx, total_loss)

@triton.jit
def fused_softcapped_entropy_bwd_kernel(
    grad_input_ptr, grad_output_ptr, lse_ptr, logits_ptr, targets_ptr, mtp_weights_ptr,
    stride_logits_n, stride_logits_v, stride_grad_n, stride_grad_v,
    n_rows, n_cols, n_predict,
    A, B, C,
    BLOCK_SIZE: tl.constexpr,
    N_PREDICT: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)

    logits_row_ptr = logits_ptr + row_idx * stride_logits_n
    grad_row_ptr = grad_input_ptr + row_idx * stride_grad_n

    lse = tl.load(lse_ptr + row_idx).to(tl.float32)
    grad_loss = tl.load(grad_output_ptr + row_idx).to(tl.float32)

    A_f32 = tl.cast(A, tl.float32)
    B_f32 = tl.cast(B, tl.float32)
    C_f32 = tl.cast(C, tl.float32)
    inv_C = 1.0 / C_f32
    B_div_C = B_f32 * inv_C
    inv_C_A = inv_C * A_f32

    # Preload all targets and weights before the column loop
    S_w = tl.cast(0.0, tl.float32)
    t0 = -1
    t1 = -1
    t2 = -1
    w0 = tl.cast(0.0, tl.float32)
    w1 = tl.cast(0.0, tl.float32)
    w2 = tl.cast(0.0, tl.float32)

    if N_PREDICT >= 1:
        if row_idx + 0 < n_rows:
            w0 = tl.load(mtp_weights_ptr + 0).to(tl.float32)
            t0 = tl.load(targets_ptr + row_idx + 0).to(tl.int32)
            S_w += w0

    if N_PREDICT >= 2:
        if row_idx + 1 < n_rows:
            w1 = tl.load(mtp_weights_ptr + 1).to(tl.float32)
            t1 = tl.load(targets_ptr + row_idx + 1).to(tl.int32)
            S_w += w1

    if N_PREDICT >= 3:
        if row_idx + 2 < n_rows:
            w2 = tl.load(mtp_weights_ptr + 2).to(tl.float32)
            t2 = tl.load(targets_ptr + row_idx + 2).to(tl.int32)
            S_w += w2

    grad_scale_icA = grad_loss * inv_C_A

    zero_f32 = tl.cast(0.0, tl.float32)
    one_f32 = tl.cast(1.0, tl.float32)

    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        val = tl.load(logits_row_ptr + cols, mask=mask, other=zero_f32).to(tl.float32)
        u = val * inv_C + B_div_C
        sigmoid_u = tl.sigmoid(u)
        z = A_f32 * sigmoid_u
        p = tl.exp(z - lse)

        term1 = S_w * p

        term2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        if N_PREDICT >= 1:
            term2 += tl.where(cols == t0, w0, zero_f32)
        if N_PREDICT >= 2:
            term2 += tl.where(cols == t1, w1, zero_f32)
        if N_PREDICT >= 3:
            term2 += tl.where(cols == t2, w2, zero_f32)

        grad_z = term1 - term2
        grad_x = grad_scale_icA * grad_z * sigmoid_u * (one_f32 - sigmoid_u)
        grad_x = grad_x.to(tl.bfloat16)
        tl.store(grad_row_ptr + cols, grad_x, mask=mask)

CE_KERNEL_BLOCK_SIZE = 256
CE_KERNEL_VOCAB_SIZE = 50304

CE_KERNEL_DECLS = f"""
constexpr int VOCAB_SIZE = {CE_KERNEL_VOCAB_SIZE};
constexpr int BLOCK_SIZE = {CE_KERNEL_BLOCK_SIZE};
"""

CE_KERNEL_SOURCE = """
#include <cuda_bf16.h>
#include <math_constants.h>

#define __nv_fp8_e5m2 char
#define uint16_t unsigned short
#define uint8_t unsigned char
#define int64_t long long

__device__ __forceinline__ __nv_fp8_e5m2 f32_to_fp8_e5m2(float x) {
    uint16_t packed;
    asm volatile(
        "cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;"
        : "=h"(packed)
        : "f"(x), "f"(0.0f)
    );
    __nv_fp8_e5m2 result;
    *reinterpret_cast<uint8_t*>(&result) = (packed & (0xFF << 8)) >> 8;
    return result;
}

struct __align__(16) __nv_bfloat168 {
    __nv_bfloat16 data[8];
    __device__ __nv_bfloat16& operator[](int i) { return data[i]; }
    __device__ const __nv_bfloat16& operator[](int i) const { return data[i]; }
};

struct __align__(8) __nv_fp8_e5m28 {
    __nv_fp8_e5m2 data[8];
    __device__ __nv_fp8_e5m2& operator[](int i) { return data[i]; }
    __device__ const __nv_fp8_e5m2& operator[](int i) const { return data[i]; }
};

template<typename T> __device__ constexpr T CEIL_DIV(T a, T b) { return (a + b - 1) / b; }

//__device__ float sigmoid(float x) {
//  return 1.0f / (1.0f + __expf(-x));
//}
__device__ float sigmoid(float x) {
  return 0.5f + __tanhf(x * 0.5f) * 0.5f;
}

extern "C"
__launch_bounds__(BLOCK_SIZE, 2)
__global__ void ce_fwd_bwd_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const float* __restrict__ mtp_weights,
    float* __restrict__ losses,
    __nv_fp8_e5m2* grad_input,
    int batch_size,
    int n_predict,
    double A_param,
    double B_param,
    double C_param,
    double grad_s_param,
    double grad_scale_param)
{
  constexpr int VEC_WIDTH = 8;
  constexpr int NUM_FULL_LOADS = VOCAB_SIZE / (BLOCK_SIZE * VEC_WIDTH);
  constexpr int NUM_LOADS = CEIL_DIV(VOCAB_SIZE, BLOCK_SIZE * VEC_WIDTH);

  float A = (float)A_param;
  float B = (float)B_param;
  float C = (float)C_param;
  float grad_s = (float)grad_s_param;
  float grad_scale = (float)grad_scale_param;

  extern __shared__ __nv_bfloat16 smem[];

  static_assert(VEC_WIDTH == 8);

  const __nv_bfloat16 *block_logit_ptr = logits + VOCAB_SIZE * blockIdx.x;

  float inv_C = 1 / C;
  float B_div_C = B * inv_C;
  float thread_max = -CUDART_INF_F;

  #pragma unroll 25
  for (int i = 0; i < NUM_LOADS; i++) {
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
      __nv_bfloat168 result = *(__nv_bfloat168*)(&block_logit_ptr[idx]);
      __nv_bfloat168 result_sigmoid;
      #pragma unroll
      for (int k = 0; k < VEC_WIDTH; k++) {
        float tmp = __bfloat162float(result[k]);
        tmp = sigmoid(tmp * inv_C + B_div_C);
        result_sigmoid[k] = __float2bfloat16(tmp);
        tmp = A * tmp;
        thread_max = max(tmp, thread_max);
      }
      *(__nv_bfloat168*)(&smem[idx]) = result_sigmoid;
    }
  }

  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  int warp_id = threadIdx.x / 32;
  __shared__ float block_maxs[NUM_WARPS];
  __shared__ float block_sums[NUM_WARPS];

  for (int offset = 16; offset > 0; offset >>= 1)
    thread_max = fmaxf(thread_max, __shfl_down_sync(0xFFFFFFFF, thread_max, offset));

  if (threadIdx.x % 32 == 0) {
    block_maxs[warp_id] = thread_max;
  }

  __syncthreads();

  float block_max = -CUDART_INF_F;
  for (int i = 0; i < NUM_WARPS; i++) {
    block_max = fmaxf(block_max, block_maxs[i]);
  }

  float thread_sum = 0.0f;
  #pragma unroll 2
  for (int i = 0; i < NUM_LOADS; i++) {
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    __nv_bfloat168 l;
    if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
      l = *(__nv_bfloat168*)(&smem[idx]);
    }
    #pragma unroll
    for (int k = 0; k < VEC_WIDTH; k++) {
      float tmp = A * __bfloat162float(l[k]);
      tmp = __expf(tmp - block_max);
      if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
        thread_sum += tmp;
      }
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1)
    thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);

  if (threadIdx.x % 32 == 0) {
    block_sums[warp_id] = thread_sum;
  }

  __syncthreads();

  float block_sum = 0.0f;
  for (int i = 0; i < NUM_WARPS; i++) {
    block_sum += block_sums[i];
  }

  float lse = block_max + __logf(block_sum);

  if (threadIdx.x == 0) {
    float total_loss = 0.0f;
    for (int k = 0; k < n_predict; k++) {
      int64_t target_idx = blockIdx.x + k;
      if (target_idx < batch_size) {
        float weight = mtp_weights[k];
        int64_t target = targets[target_idx];
        if (target >= 0 && target < VOCAB_SIZE) {
          float z_target = A * __bfloat162float(smem[target]);
          total_loss += weight * (lse - z_target);
        }
      }
    }
    losses[blockIdx.x] = total_loss;
  }

  float S_w = 0.0f;

  for (int i = 0; i < n_predict; i++) {
    S_w += mtp_weights[i];
  }

  #pragma unroll 4
  for (int i = 0; i < NUM_LOADS; i++) {
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    __nv_fp8_e5m28 result;

    if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
      __nv_bfloat168 sigmoid_us = *(__nv_bfloat168*)(&smem[idx]);
      #pragma unroll
      for (int j = 0; j < VEC_WIDTH; j++) {
        float sigmoid_u = __bfloat162float(sigmoid_us[j]);
        float z = A * sigmoid_u;
        float p = __expf(z - lse);

        float term1 = S_w * p;
        float term2 = 0.0f;

        float grad_z = term1 - term2;
        float grad_x = grad_scale * (1.0f / C * A) * (1.0f / grad_s) * grad_z * sigmoid_u * (1.0f - sigmoid_u);
        auto result_tmp = f32_to_fp8_e5m2(grad_x);
        result[j] = *reinterpret_cast<__nv_fp8_e5m2*>(&result_tmp);
      }
      *(__nv_fp8_e5m28*)(&grad_input[blockIdx.x * VOCAB_SIZE + idx]) = result;
    }
  }

  __syncthreads();

  if (threadIdx.x < n_predict && blockIdx.x + threadIdx.x < batch_size) {
    int i = threadIdx.x;
    int64_t target = targets[blockIdx.x + i];

    float sigmoid_u = __bfloat162float(smem[target]);
    float z = A * sigmoid_u;
    float p = __expf(z - lse);

    float term1 = S_w * p;
    float term2 = 0.0f;

    #pragma unroll
    for (int k = 0; k < 3; k++) {
      int64_t target_idx = blockIdx.x + k;
      if (target_idx < batch_size && k < n_predict) {
        if (targets[target_idx] == target) {
          term2 += mtp_weights[k];
        }
      }
    }

    float grad_z = term1 - term2;
    float grad_x = grad_scale * (1.0f / C * A) * (1.0f / grad_s) * grad_z * sigmoid_u * (1.0f - sigmoid_u);
    auto result_tmp = f32_to_fp8_e5m2(grad_x);
    auto result = *reinterpret_cast<__nv_fp8_e5m2*>(&result_tmp);
    grad_input[blockIdx.x * VOCAB_SIZE + target] = result;
  }
}
"""

_ce_fwd_bwd_kernel = None

def get_ce_fwd_bwd_kernel():
    global _ce_fwd_bwd_kernel
    if _ce_fwd_bwd_kernel is None:
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        cuda_include_dirs = [os.path.join(cuda_home, "include")] if cuda_home else ["/usr/local/cuda/include/"]
        _ce_fwd_bwd_kernel = torch.cuda._compile_kernel(
            CE_KERNEL_DECLS + CE_KERNEL_SOURCE,
            "ce_fwd_bwd_kernel",
            compute_capability="90",
            cuda_include_dirs=cuda_include_dirs,
            nvcc_options=["-lineinfo", "--use_fast_math"],
        )
        _ce_fwd_bwd_kernel.set_shared_memory_config(CE_KERNEL_VOCAB_SIZE * 2)
    return _ce_fwd_bwd_kernel

@torch.library.custom_op("nanogpt::ce_fwd_bwd", mutates_args={"losses", "grad_input"})
def ce_fwd_bwd(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mtp_weights: torch.Tensor,
    losses: torch.Tensor,
    grad_input: torch.Tensor,
    n_rows: int,
    n_predict: int,
    A: float,
    B: float,
    C: float,
    grad_s: float,
    grad_scale: float,
) -> None:
    if not use_fp8():
        raise RuntimeError("ce_fwd_bwd requires Hopper (sm_90+); should be unreachable when use_fp8() is False.")
    kernel = get_ce_fwd_bwd_kernel()
    grid = (n_rows, 1, 1)
    kernel(
        grid,
        (CE_KERNEL_BLOCK_SIZE, 1, 1),
        (logits, targets, mtp_weights, losses, grad_input,
         n_rows, n_predict, A, B, C, grad_s, grad_scale),
        shared_mem=CE_KERNEL_VOCAB_SIZE * 2,
    )

class FusedSoftcappedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, targets, mtp_weights, lm_head_weight, x_s, w_s, grad_s, grad_scale, A=23.0, B=5.0, C=7.5):
        if not use_fp8():
            # BF16 fallback for GPUs without col-major FP8 _scaled_mm (e.g. L40)
            logits = (x @ lm_head_weight).bfloat16()
            n_rows, n_cols = logits.shape
            if mtp_weights is None:
                 mtp_weights = torch.tensor([1.0], device=logits.device, dtype=torch.float32)
            n_predict = mtp_weights.shape[0]
            losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
            lse = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
            logits = logits.contiguous()
            targets = targets.contiguous()
            mtp_weights = mtp_weights.contiguous()
            grid = (n_rows,)
            fused_softcapped_entropy_fwd_kernel[grid](
                logits, losses, lse, targets, mtp_weights,
                logits.stride(0), logits.stride(1),
                n_rows, n_cols, n_predict,
                A, B, C,
                BLOCK_SIZE=2048,
                num_warps=2
            )
            ctx.save_for_backward(targets, mtp_weights, lse, x, lm_head_weight)
            ctx.params = (A, B, C, x_s, w_s, grad_s)
            return losses

        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = lm_head_weight.div(w_s).to(torch.float8_e4m3fn)

        w_f8_col_major = w_f8.T.contiguous().T

        logits = torch._scaled_mm(
            x_f8,
            w_f8_col_major,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )

        n_rows, n_cols = logits.shape
        if mtp_weights is None:
             mtp_weights = torch.tensor([1.0], device=logits.device, dtype=torch.float32)
        n_predict = mtp_weights.shape[0]

        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        lse = torch.empty(n_rows, dtype=torch.float32, device=logits.device)

        logits = logits.contiguous()
        targets = targets.contiguous()
        mtp_weights = mtp_weights.contiguous()

        grad_input = torch.empty((n_rows, n_cols), dtype=torch.float8_e5m2, device=logits.device)

        ce_fwd_bwd(logits, targets, mtp_weights, losses, grad_input,
             n_rows, n_predict, A, B, C, grad_s, grad_scale)

        ctx.save_for_backward(logits, targets, mtp_weights, lse, x, lm_head_weight, x_f8, w_f8, grad_input)
        ctx.params = (A, B, C, x_s, w_s, grad_s)
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        if not use_fp8():
            targets, mtp_weights, lse, x, lm_head_weight = ctx.saved_tensors
            A, B, C, x_s, w_s, grad_s = ctx.params
            n_rows = x.shape[0]
            n_cols = lm_head_weight.shape[1]  # vocab_size
            n_predict = mtp_weights.shape[0]
            # BF16 fallback: recompute logits here (not saved in ctx to avoid holding
            # a ~4.7 GiB tensor).
            logits = (x @ lm_head_weight).bfloat16().contiguous()
            grad_input = torch.empty((n_rows, n_cols), dtype=torch.bfloat16, device=logits.device)
            grid = (n_rows,)
            fused_softcapped_entropy_bwd_kernel[grid](
                grad_input, grad_output.contiguous(), lse, logits, targets, mtp_weights,
                logits.stride(0), logits.stride(1), grad_input.stride(0), grad_input.stride(1),
                n_rows, n_cols, n_predict,
                A, B, C,
                BLOCK_SIZE=1024,
                num_warps=4,
                N_PREDICT=n_predict,
            )
            del logits  # free ~4.7 GiB before the grad matmuls
            grad_x = grad_input @ lm_head_weight.T
            grad_w = x.T @ grad_input
            return grad_x.bfloat16(), None, None, grad_w, None, None, None, None

        logits, targets, mtp_weights, lse, x, lm_head_weight, x_f8, w_f8, grad_input = ctx.saved_tensors
        A, B, C, x_s, w_s, grad_s = ctx.params
        n_rows, n_cols = logits.shape
        n_predict = mtp_weights.shape[0]

        grad_output = grad_output.contiguous()

        x_scale = grad_input.new_tensor(x_s, dtype=torch.float32)
        w_scale = grad_input.new_tensor(w_s, dtype=torch.float32)
        grad_scale = grad_input.new_tensor(grad_s, dtype=torch.float32)

        grad_x = torch._scaled_mm(
            grad_input,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=grad_scale,
            scale_b=w_scale,
            use_fast_accum=False,
        )

        x_f8_T = torch.empty((x_f8.shape[1], x_f8.shape[0]), dtype=x_f8.dtype, device=x_f8.device)
        transpose_copy(x_f8, x_f8_T)  # (768, n_rows) row-major

        grad_input_T = torch.empty((n_cols, n_rows), dtype=grad_input.dtype, device=grad_input.device)
        transpose_copy(grad_input, grad_input_T)  # (50304, n_rows) row-major

        grad_w = torch._scaled_mm(
            x_f8_T,            # (768, n_rows) row-major
            grad_input_T.T,    # (n_rows, 50304) column-major view
            out_dtype=torch.float32,
            scale_a=x_scale,
            scale_b=grad_scale,
            use_fast_accum=False,
        )

        return grad_x, None, None, grad_w, None, None, None, None

