# Modded-NanoGPT (Lovelace fork)

This is a fork of [upstream
modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) ported to run
on non-Hopper-class architectures. Specifically it has been tested on L40 GPUs
(Lovelace architecture) on the American University [HPC
cluster](https://www.american.edu/cas/hpc/) (coincidentally also named
Lovelace).

Main changes:
* When Flash Attention 3 is unavailable, fall back to pure PyTorch
  scaled-dot-product-attention (SDPA).
* Add BF16 Triton kernels as fallbacks for kernels that assume FP8, TMA, and/or
  sm90 inline assembly.
* Add --val-batch-size CLI argument to support lower VRAM usage.

Note: only `train_gpt.py` has been ported, not `train_gpt_medium.py`.

See the upstream README for the rest of the info!
