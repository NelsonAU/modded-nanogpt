# Modded-NanoGPT fork for class demos

[Mark J. Nelson](https://www.kmjn.org/), 2026

This is a fork of [upstream
modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) that was split
off in May 2026. The main original change was porting it to run on a broader
set of Nvidia GPU architectures. Upstream assumed Hopper-class GPUs,
specifically H100; this fork also runs on Ampere and Ada Lovelace
architectures.

I've also added some niceties for the purpose of class demos, like printing
sample generated text as the training run progresses, to give a qualitative
idea of how generation improves during training.

Since my fork, upstream has doubled down even more on speedrunning
*specifically* the setting of 8xH100 GPUs, so I have not continued to track
upstream commits. But I do still think `modded-nanogpt` as of sometime in
mid-2026 was a good starting point for this kind of demo, because many of its
architectural modernizations over the original GPT-2 and over Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT/) (rotary embeddings, ReLU², Muon
optimizer, skip connections, etc.) are more broadly useful.

Main changes:
* When Flash Attention 3 is unavailable, fall back to pure PyTorch
  scaled-dot-product-attention (SDPA).
* Add a mix of pure PyTorch fallbacks and BF16 Triton kernels in place of
  kernels that assume FP8, TMA, and/or sm90 inline assembly.
* Choose validation batch size based on GPU vRAM (upstream setting was only
  suitable for 80GB+ cards).
* Add sample text generation during training validation passes and
  countdown ETA to the next validation check.

Note: only `train_gpt.py` has been ported, not `train_gpt_medium.py`.
