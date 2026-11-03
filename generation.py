"""
Text generation helpers for periodic sample printing during training.

Usage pattern (mirrors evals/hellaswag.py):
    from generation import _GEN_PROMPTS, print_samples
    # setup (once, before training loop)
    _enc = tiktoken.get_encoding("gpt2")
    _gen_prompt_ids = [[BOS_ID] + _enc.encode(p) for p in _GEN_PROMPTS[:n]]
    # inside validation block
    if master_process and args.num_gen_samples > 0:
        print_samples(_raw_model, forward_args, _gen_prompt_ids, _enc,
                      step, train_steps, max_new_tokens=..., temperature=..., top_k=...,
                      device=device, get_bigram_hash=get_bigram_hash, print0=print0)
"""
import math
import torch
import torch.nn as nn

BOS_ID = 50256

_GEN_PROMPTS = [
    "The quick brown fox",
    "Scientists recently discovered",
    "In the beginning",
    "The president announced",
    "Once upon a time",
]

@torch.no_grad()
def generate_text(raw_model: nn.Module, forward_args, prompt_ids: list[int], *,
                  max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 40,
                  device, get_bigram_hash) -> str:
    """
    Autoregressively sample tokens from the model given a prompt.
    Uses the uncompiled model (raw_model) to avoid torch.compile shape recompilations.
    Returns the decoded string (prompt + generated continuation).
    """
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    tokens = list(prompt_ids)
    max_num_docs = 128  # seqlens padding size; only 1 real document boundary

    raw_model.eval()
    for _ in range(max_new_tokens):
        real_len = len(tokens)
        # Attention kernel requires T % 16 == 0; pad with BOS_ID
        T_padded = math.ceil(real_len / 16) * 16
        padded = tokens + [BOS_ID] * (T_padded - real_len)
        input_seq = torch.tensor(padded, dtype=torch.int32, device=device)

        # cu_seqlens: one document of length T_padded, rest padded with T_padded
        seqlens = torch.full((max_num_docs,), T_padded, dtype=torch.int32, device=device)
        seqlens[0] = 0

        bigram_cpu = get_bigram_hash(input_seq.cpu())
        bigram_gpu = bigram_cpu.to(device=device)

        # target_seq=None triggers generation mode → returns logits (1, T_padded, vocab_size)
        logits = raw_model(input_seq, None, seqlens, bigram_gpu, forward_args)
        next_logits = logits[0, real_len - 1, :].float()  # logit at last *real* position

        next_logits = next_logits / max(temperature, 1e-8)
        if top_k > 0:
            top_vals = torch.topk(next_logits, min(top_k, next_logits.size(-1))).values
            next_logits[next_logits < top_vals[-1]] = float("-inf")
        probs = torch.softmax(next_logits, dim=-1)
        next_token = int(torch.multinomial(probs, num_samples=1).item())
        tokens.append(next_token)
        if next_token == BOS_ID:  # EOS / document boundary → stop
            break

    # Decode, stripping BOS/pad tokens
    decode_ids = [t for t in tokens if t != BOS_ID]
    return enc.decode(decode_ids)


def print_samples(raw_model: nn.Module, forward_args, prompt_ids_list: list[list[int]], enc,
                  step: int, train_steps: int, *,
                  max_new_tokens: int, temperature: float, top_k: int,
                  device, get_bigram_hash, print0) -> None:
    """Print generated text samples; call from within the validation block."""
    sep = "─" * 72
    print0(sep, console=True)
    print0(f"  Generated samples at step {step}/{train_steps} "
           f"(temp={temperature:.2f}, top_k={top_k})", console=True)
    print0(sep, console=True)
    for i, prompt_ids in enumerate(prompt_ids_list):
        prompt_text = enc.decode([t for t in prompt_ids if t != BOS_ID])
        try:
            sample = generate_text(
                raw_model, forward_args, prompt_ids,
                max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
                device=device, get_bigram_hash=get_bigram_hash,
            )
        except Exception as e:
            sample = f"[generation error: {e}]"
        print0(f"  [{i+1}] prompt: \"{prompt_text}\"", console=True)
        print0(f"       → {sample}", console=True)
    print0(sep, console=True)
