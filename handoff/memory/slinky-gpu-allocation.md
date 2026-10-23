---
name: slinky-gpu-allocation
description: On node slinky-0 the user first restricted work to GPUs 6,7, then (2026-10-23, later) allowed all 8 GPUs
metadata:
  type: feedback
---

On node slinky-0 the user first asked for GPU work on GPUs 6 and 7 only ("Please use gpu6-7 to do your works", 2026-10-23). Later the same day they allowed all 8 GPUs ("你可以用8GPUs 做") for finding further speed-ups.

**Why:** the other GPUs were presumably in use earlier; by the evening of 2026-10-23 all 8 were idle (nvidia-smi 0 MiB used).
**How to apply:** check `nvidia-smi` before using GPUs 0-5; default to GPUs 6,7 for small kernel work (`CUDA_VISIBLE_DEVICES=6,7`) and use all 8 (torchrun) only for multi-GPU measurements. See [[xllm-gekko-step-profile]] for the 8-GPU FSDP harness.
