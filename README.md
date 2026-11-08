# Modded-NanoGPT fork for class demos

[Mark J. Nelson](https://www.kmjn.org/), 2026

This is a fork of [upstream
modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) that I split
off in May 2026, intended for use as a class demo where we train a (small) LLM
from scratch live and discuss what's going on.

When I first ran across Keller Jordan's `modded-nanogpt` in early 2026, and the
"speedrun" community that had developed around it, I was surprised that you
could now train a GPT-2-small model in 90 seconds! That did come with one big
asterisk: *on eight H100 GPUs*, which is something like $250,000 of GPU. A bit
more than I have lying around in my classroom. But it got me thinking: if they
can do it in 90 seconds on that hardware, surely we can do it within an hour on
cheaper hardware, fast enough for a single-class-period demo? Which did turn
out to be the case.

The main change of this fork is that it only requires CUDA compute
capability >=8.0 (Ampere or newer), while upstream requires >=9.0 (Hopper).
Upstream also implicitly assumes ample VRAM in several places where it's not
really necessary (H100s have 80GB), and this port scales that to allow common
24GB consumer cards like the 3090 and 4090 to run it.

I've also added some niceties for the purpose of class demos, like periodically
sampling text from the model as the training run progresses, to give a
qualitative idea of how generation improves during training. This part is
something of a work in progress in terms of how nice and how configurable the
output is, but all the key parts are now there (`generation.py` is new in this
fork).

*Relationship to upstream:* Since my fork, the competitive pressure of speedruns
has made the upstream code base progressively more tied to *specifically* the
setting of 8xH100 GPUs. This has brought the time down even further, to 40
seconds! But it also now hardcodes an 8x GPU assumption, has a 130GB n-gram
table in VRAM, has huge startup latency to move everything possible out of the
official timed segment, etc. So this repo no longer tracks upstream, besides
occasionally cherry-picking a bugfix/modernization. I do still think that
`modded-nanogpt` as of sometime in early/mid-2026 was a good starting point for
this kind of demo, because many of its original speedups over the original
GPT-2 and over Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT/) were
more architectural and broadly useful (rotary embeddings, ReLU², Muon
optimizer, skip connections, etc.).

Main changes from upstream:
* Add live progress: periodically generates sample text to show qualitative
  progress, along with intermediate validation loss calculation for
  quantitative progress.
* When Flash Attention 3 is unavailable, fall back to PyTorch flex_attention
  with sliding-window block masking.
* For Ampere and Ada Lovelace architectures, add a mix of pure PyTorch
  fallbacks and BF16 Triton kernels in place of kernels that assume FP8, TMA,
  and/or sm90 inline assembly.
* Choose validation batch size based on GPU VRAM (upstream setting is only
  suitable for 80GB+ cards).
* Scale gradient accumulation steps based on GPU VRAM, which lets us support
  24GB GPUs (e.g. RTX 3090/4090).

## Tested cards

This has been tested to run on the following cards, with training-only timings
as follows. The 8xH100 timing is the upstream speedrun; the rest are my tests
as of November 2026. The tested set spans consumer, workstation, and datacenter
cards from the Ampere, Ada Lovelace, and Hopper architectures.

* 8xH100: 1.5 minutes
* 4xA100-SXM4: 10 minutes
* 1xH200: 12 minutes
* 4xL40: 18 minutes
* 1xA100-SXM4: 32 minutes
* 1xGeForce RTX 4090: 50 minutes (70 minutes with flex_attention fallback)
* 1xL40: 66 minutes
* 1xA40: 69 minutes

Total wall-clock time for a classroom demo will be a bit longer due to time
needed for the intermediate text generation. How much longer depends on how
frequently you sample (default is every 25 training steps) and on what
hardware. As approximate timings, it adds ~2-10 seconds per sample set. With
the default every-25-steps sampling frequency, this adds ~2-10 minutes total.

## Validation plot

A sanity check that all the different paths (upstream sm90 assembly, my fallbacks, etc.) on various hardware all produce the same result, albeit at different speeds:

<img width="3000" height="1800" alt="val_loss_vs_steps" src="https://github.com/user-attachments/assets/66fc0454-b03e-4a8c-afbc-55e472d12c52" />
