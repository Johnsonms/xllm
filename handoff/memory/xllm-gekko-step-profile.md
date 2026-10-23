---
name: xllm-gekko-step-profile
description: Gekko training-step profile on B200 (2026-10-23) - where the time goes, measured speed-up candidates, harness scripts
metadata:
  type: project
---

Full-step profiles (1 GPU, L=16384, torch.profiler + chrome-trace call-site attribution) taken 2026-10-23 after the CuTe AWM backward landed. Step is GPU-bound; FSDP2 on 8 GPUs (torchrun) hides all NCCL time (8-GPU fwd+bwd == 1-GPU), so kernels are the only lever on one node.

**gekko-0.9B B2 (1258 ms/step, cub TSDN):** swift sliding-chunk attention (SCA) fwd+bwd 671 ms = 58% (cuBLAS batched GEMMs + materialized bf16 attention matrix + ~300 ms of contiguous copies: `attn_w[...] = w`, repeated K/V, split w in bwd); attention matrix alone ~105 GB of the 139 GB peak. TSDN CUB ~120 ms; cross-entropy grad GEMM `torch.mm(scores, weight_fp32)` fp32 SIMT 64 ms (train.py enables TF32 -> ~5 ms there); rope cat/complex kernels ~70 ms; norms ~65 ms; AWM cute ~10 ms.
**gekko-7.1B B1 (1150 ms/step):** GEMMs 580-610 ms (54%, ~1.3 PFLOPS achieved vs 1.6 PFLOPS cuBLAS standalone on the same shapes); SCA 265 ms (23%); TSDN CUB ~130 ms; fp32 CE grad GEMM 67 ms; AWM cute ~23 ms.
**xattn (fast flash SCA used on H200) does not support SM100** -> a CuTe DSL sliding-chunk attention for Blackwell is the #1 item (also frees ~100 GB -> production batch sizes).
**timenorm_backend='cute' verified in-model:** loss/grad parity with cub to bf16 noise; step 0.9B 1258->1153 ms, 7.1B 1148->1050 ms (-8.5%).

**How to apply:** scripts in scratchpad `prof/` of session d3839bfb (prof_step.py = category breakdown, prof_stack2.py + attrib.py = call-site attribution via chrome trace incl. cuda_driver launches, fsdp_step.py = torchrun FSDP2 harness with clip+optim, parity.py = cub vs cute TSDN). Pitfall: torch.profiler kernel events have no cpu_parent; attribute via trace `correlation` of cuda_runtime AND cuda_driver (cuLaunchKernelEx) events, python frames are named `xllm/...py(line): fn`. Related: [[xllm-cute-awm]], [[xllm-cute-tsdn]], [[slinky-gpu-allocation]].
