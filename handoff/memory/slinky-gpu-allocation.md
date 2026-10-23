---
name: slinky-gpu-allocation
description: On node slinky-0 use only GPUs 6 and 7 (CUDA_VISIBLE_DEVICES=6,7) for all kernel work
metadata:
  type: feedback
---

On the node slinky-0 (the node the CuTe kernel work moved to on 2026-10-23), run all GPU work on GPUs 6 and 7 only: set `CUDA_VISIBLE_DEVICES=6,7` (or `6`) for tests, benchmarks, ncu and the Gekko harness.

**Why:** the user asked for it ("Please use gpu6-7 to do your works"); the other GPUs are presumably in use by others.
**How to apply:** prefix every python/pytest/ncu invocation with `CUDA_VISIBLE_DEVICES=6,7`; the harness `run_gekko.py` calls `torch.cuda.set_device(0)`, which then maps to GPU 6. See [[xllm-cute-awm]] for the environment on this node.
