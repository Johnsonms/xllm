# Porting the B200 optimizations to H100: plan for a Claude Code session on a Hopper node

This is the list of everything that was done for Gekko training on B200 (branches `cute-timestep-decay-norm`,
`cute-adaptive-working-memory`, hand-off `cute-kernels-handoff` of github.com/Johnsonms/xllm; details in
`HANDOFF.md` sections 4 to 6c), with what each item bought, which parts carry over to H100 unchanged, and what
has to be redone for sm_90a. Work through it in order: **baseline first, then one item at a time, each with
its own measurement, parity check and commit.** Push finished commits to `origin` on a new branch
(suggested name `hopper-kernels`, based on `cute-adaptive-working-memory`). Commit convention:
`git -c user.name="Johnson" -c user.email="johnson@together.ai" commit ...` with
`Co-Authored-By: Claude <noreply@anthropic.com>` as the last line. Keep a `handoff/HANDOFF_H100.md` the same
way `HANDOFF.md` was kept here.

## Kernel inventory: everything that was (re)written on B200

All CuTe Python DSL unless noted; all in `xllm/modules/fused_ops/cute/` on branch `cute-adaptive-working-memory`.
"B" marks code that uses Blackwell-only features (tcgen05 MMA, TMEM, packed f32x2 math) and needs a Hopper
rewrite; "P" marks code that should port with small changes.

| Kernel / class | File | Replaces | Role, structure | B200 effect | Port |
|---|---|---|---|---|---|
| `TimestepDecayNormFwd` / `Bwd` | `timestep_decay_norm.py` | CUB `timenorm_backend='cub'` | single-pass decoupled look-back scan over time, TMA tiles, aggregate / resolve warps | fwd 102 vs 1488 us, bwd 156 vs 3104 (B8 L8192 H2048 G32), ~85% HBM SOL | P (packed f32x2 -> scalar; retune T, warps) |
| `AwmKeyLsePart`, `AwmChunkGemm`, `AwmAToBf16`, `AwmRecurrence` / `AwmRecurrenceUmma`, `AwmOutGemm` | `adaptive_working_memory.py` | torch AWM forward | phase decomposition (A|B chunk GEMMs in parallel, S x V recurrence, out GEMM); masks from segment_idx in kernel | fwd 1376 -> 125 us (0.9B), 729 -> 91 (4B), 1259 -> 207 (7.1B) | B for the tcgen05 GEMMs (ChunkGemm, OutGemm, RecurrenceUmma); P for KeyLsePart, AToBf16, the warp-mma AwmRecurrence |
| `AwmDmoGemm`, `AwmRecurrenceBwd`, `AwmBwdDelta`, `AwmLseChain`, `AwmChunkBwd`, `AwmZeroTail` (+ dq via `AwmOutGemm`) | `adaptive_working_memory_bwd.py` | torch AWM backward | the backward phases of `adaptive_working_memory_reference.py` (executable spec), workspaces saved by the forward | bwd per step 0.9B 99 -> 6 ms, 7.1B 98 -> 11 ms | B (DmoGemm, ChunkBwd are tcgen05); P for the rest |
| `ScaFwd` | `sliding_chunk_attention.py` | swift SCA (cuBLAS + materialized attention matrix) | flash-style forward, q in smem, P in TMEM as the PV A operand, lazy rescale | 0.9B 0.83 ms/layer vs ~12 | B (FA3 structure on Hopper) |
| `ScaFwdT` (default) | same | `ScaFwd` | q tile in TMEM as the A operand; 4B: 256 TMEM cols, two CTAs per SM | 4B 668 -> 550 us per call, 0.9B 558 -> 540 | B (no TMEM on Hopper: nothing to port, FA3 keeps Q in smem / registers) |
| `ScaBwdDv`, `ScaBwdDk`, `ScaBwdDq` | same | swift backward | generic backward (3 kernels, v slabs for V > 256, fp32 dP from cuBLAS for V > 256) | 0.9B bwd 3.0 ms/layer vs ~11 | B |
| `ScaBwdDkDv`, `ScaBwdDqT` (default for S + V <= 256) | same | `ScaBwdDv` + `ScaBwdDk`, `ScaBwdDq` | fused dK+dV, K/V/Q/dO in TMEM as A operands, 2 compute groups, raw mbarriers | 4B bwd 3.6 -> 1.8 ms/layer | B (the FA3 backward is the Hopper equivalent; pipeline order and the predication pitfall transfer) |
| `ScaPmat`, `ScaDsMat` + cuBLAS bmm (default for V > 256 when > 50% of the window is attended) | same | `ScaBwdDv` v-slab path, `ScaBwdDk`, `ScaBwdDq` | materialized bf16 P and dS = P (dP - delta) (chunk-major [C x 2C]); dV, dK, dQ as two bmm per chunk each; chosen per call from qmax | 7.1B backward 6.13 -> 4.16 ms/layer (no boundaries) | B for ScaPmat (q in TMEM); ScaDsMat is a plain memory kernel (P); the cuBLAS path is portable as is |
| `ChunkMajorCopy`, `ChunkMajorUncopy`, `RowDot` | same | torch permute copies / fp32 casts | layout copies (~1.7 TB/s in torch) and delta = rowsum(dO * y) | ~2.4 ms/layer saved on 7.1B | P (plain memory kernels) |
| host: `segment_starts` / `query_limits` (cached), `_use_pmat_dv` | same | - | per-row key ranges from segment_idx; path choice with one sync | - | P |
| `ScaBwdDvT`, `ScaBwdDkT`, `ScaBwdDqT(q_smem)` and their `load_dp` modes (default for S + V > 256, incl. the 7.1B) | same | `ScaBwdDv`, `ScaBwdDk`, `ScaBwdDq` | TMEM-A variants with one operand left in smem (or dP read from the GEMMs) to fit 512 columns; 32- or 64-wide stages | 0.9B bwd 2972 -> 2415 us/layer; 7.1B kernel-path bwd 3394 -> 2766 | B (nothing to port: on Hopper all operands are in smem anyway; the FA3 backward structure applies) |
| `_bmm_tril_*`, `_bmm_triu_nt`, `_compile_bwd_paths`, DP threshold | same | dense cuBLAS GEMMs over causal blocks | 4 x 4 sub-blocked bmm; both paths compiled at the first call; GEMM path from fraction 0.35 under data parallelism | 7.1B bwd 4.16 -> 3.76 ms/layer; 8 GPUs 1017 -> 884 ms | P (host code, portable as is) |
| `Rope` | `rope.py` | `apply_rope` / `apply_ropes` (torch, 4-5 passes) | one bf16 pass, fp32 math from the (cos, sin) table | 0.9B step 527 -> 469 ms | P (plain memory kernel; any CUDA / Triton version works) |
| `QkPrepFwd` / `QkPrepBwd` | `qk_prep.py` | rmsnorm + rope + fp32 softmax and their backward | warp per head row, partials in registers | 0.9B 433 -> 379, 4B 381 -> 368 | P |
| `AttnGateFwd` / `Bwd` | `attn_gate.py` | add + rmsnorm + softplus gate + mul and backward | 1-8 warps per row, torch roundings reproduced | 0.9B 379 -> 356, 4B 368 -> 334, 7.1B 893 -> 852 | P |
| `SwigluFwd` / `Bwd` | `swiglu.py` | silu * h3; recompute + silu_backward + 2 muls | 3 / 6 passes | 7.1B 813 -> 798 | P |
| `CeLogits` | `cross_entropy.py` | ~12 torch ops per logits chunk | CTA per row: loss + fp32 / bf16 scores | 0.9B 352 -> 338 | P |
| `LnFwd` / `LnBwd` | `layer_norm.py` | extension GroupLayerNorm | C / 256 warps per row | 0.9B 348 -> 335, 7.1B 850 -> 826 | P |
| `ConvFwd` (`cute/causal_conv.py`, forward only) | `causal_conv1d.py` dispatch (backend 'triton') | Triton causal conv1d forward | thread = 8 channels x 4-row steps, 3-row register window, packed f32x2, mask-free path | 0.9B q 128 -> 75 us, v 126 -> 59; 7.1B v 207 -> 164 | P (plain memory kernel; on H100 compare with the Triton kernel first - it may already be at bandwidth there) |
| csrc `GammaBwdLargeKernel` / `GammaBetaBwdLargeKernel` | `xllm/csrc/ops/group_*_norm_kernel.cu` | the same kernels (32-64 blocks) | row-split, 16-byte loads, fp32 partials + finalize | 0.9B 469 -> 433, 7.1B 924 -> 910 | P (identical CUDA; rebuild the extension) |

Not kernels, but part of the B200 result (section 2): the `create_on_gpu` default-device fix, `fsdp_layers_per_group`,
the profiling scripts.

## 0. What H100 changes, in one paragraph

B200 (sm_100a) has 5th-gen tensor cores driven by `tcgen05` with accumulators in **tensor memory (TMEM)**, ~2.25
PFLOPS dense bf16, 8 TB/s HBM3e, 148 SMs x 228 KB smem, and 180 GB per GPU. H100 SXM (sm_90a) has `wgmma`
(accumulators in **registers**, operands from smem / registers), ~1 PFLOPS dense bf16, 3.35 TB/s HBM3, 132 SMs
x 228 KB smem, 80 GB. Everything that was TMEM / tcgen05 specific (the AWM GEMM kernels, the SCA kernels, the
"A operand in TMEM" trick of §6b) must be re-expressed FA3-style: TMA producer warp + 2 consumer warpgroups,
`wgmma` with register accumulators, P kept in registers (or smem) between the two GEMMs, `setmaxnreg` to give
the consumer warpgroups ~240 registers. Everything that was memory-bound or host-side carries over.
Reference code on Hopper: FlashAttention-3 (`flash_attn/hopper`, CUTLASS C++) and the CuTe DSL Hopper examples
(`cutlass/examples/python/CuTeDSL/cute/hopper/`: dense GEMM with TMA + wgmma warp specialization). The CuTe
Python DSL (nvidia-cutlass-dsl) supports sm_90a; check the installed version's Hopper coverage first
(`cute.nvgpu.warpgroup`, `cpasync`, `pipeline.PipelineTmaAsync`).

## 1. Environment and baseline (do this first, do not skip)

1. Node check: `nvidia-smi` (H100 SXM vs PCIe, HBM size), NVLink topology (`nvidia-smi topo -m`), CUDA /
   driver, whether `ncu` is available (on B200 it was not installed system-wide: `apt-get download
   nsight-compute-* && dpkg -x` into scratch worked without sudo). Build the `xllm_extension` (`setup.py
   build_ext`) against the node's CUDA; HANDOFF.md §2 has the pip-toolchain pitfalls.
2. venv: torch (cu12x for Hopper), `nvidia-cutlass-dsl`, `apache-tvm-ffi`, the harness deps
   (`handoff/venv-freeze.txt` lists what the B200 venv had). Check whether **xattn** (`xattn.ops`, the flash
   SCA used on H200) installs on this node: it is the production SCA backend on Hopper (`sca_backend='xattn'`)
   and did not support SM100, which is why the CuTe SCA was written at all.
3. Copy `handoff/scripts/profile/` and run the same harness as on B200 (all numbers below come from it):
   - `run1.sh <gpus> <nproc> <port> <trace> <preset> <B> <L> [steps]` -> `fsdp_trace.py` (torchrun, FSDP2,
     clip + optimizer, chrome trace of one step with the `FWDBWD` window; env `CFG_<field>=value` overrides
     any ModelConf field, `AWM_BACKEND`, `TN_BACKEND`, `SAME_SEED`, `SEED_OFFSET`, `PROF_STEP`, `WITH_STACK`,
     `FSDP_FP32_RS`, `FSDP_RESHARD`).
   - `gaps.py <trace>` (compute-stream idle time, NCCL overlap), `gapattr.py` / `cpuself.py` (GPU-idle time and
     CPU self time by python frame, needs `WITH_STACK=1`), `kcmp.py <a> <b>` (kernel diff of two steps),
     `prof_step.py` (kernel time by category), `prof_stack2.py` + `attrib.py` (kernel time by xllm call site).
   - Presets: gekko-0.9B (B2 L16384), gekko-4B (B1 L8192), gekko-7.1B (B1 L16384). On 80 GB the 7.1B / 0.9B
     shapes may not fit with the swift SCA (it materializes the attention matrix: ~105 GB at 0.9B B2 L16384);
     record OOMs as results and use the largest shape that fits for the baseline of each backend.
4. Baseline table to produce before touching any kernel (1 GPU and 8 GPUs, fwd+bwd ms, peak memory):
   - `timenorm_backend` cub (default) ; `awm_backend` torch (default) ; `sca_backend` xattn and swift.
   - Per-op microbenchmarks with the scripts in `handoff/scripts/{tsdn,awm,sca}` (they measure GPU time per
     call at the real shapes; the cute variants will not run on sm_90 yet, use them for the torch / cub sides).
   - The step profile breakdown (prof_step.py + attribution) so that the H100 priority order is based on H100
     numbers, not on the B200 ones below. Expect the balance to differ: H100 has 2.2x less tensor throughput
     and 2.4x less HBM bandwidth than B200, so GEMM-bound parts grow relative to host-bound ones.
5. Straggler check (§6c finding, arch independent): run 8 GPUs with `SAME_SEED=1` and without; the difference is
   data-dependent compute (SCA cost depends on the segment structure of each rank's batch), not communication.

## 2. Items that carry over unchanged (verify, measure, no port needed)

These are already on `cute-adaptive-working-memory`; they are not Blackwell specific. Keep them in the H100
baseline and just confirm their effect.

| Item | Where | B200 effect | What to check on H100 |
|---|---|---|---|
| `create_on_gpu()` restored the default device with `set_default_device("cpu")`, leaving a `DeviceContext` TorchFunctionMode active for the whole run (python hook on every torch op) | `xllm/utils.py`, commit 671884b | 4B 1-GPU 554 -> 444 ms, 0.9B 590 -> 530; 8 GPUs 4B 614 -> 504 | Measure with/without (`git stash` the 5-line change): H100 is slower per kernel, so the launch-bound share shrinks, but the mode cost is pure CPU and stays |
| `ModelConf.fsdp_layers_per_group` (several GekkoBlocks per `fully_shard` group, layer 0 alone) | `xllm/config.py`, `xllm/models/gekko.py`, commit 1f9d3c0 | 4B 8 GPUs 504 -> 485 (same batches 449 -> 430); 0.9B / 7.1B no gain | Only helps launch-bound models; on H100 with 900 GB/s NVLink the exposed first all-gather / last reduce-scatter are larger, so also re-test `reshard_after_forward=False` and `fp32_reduce_scatter=False` (no effect on B200) |
| SCA `segment_starts` / `query_limits` cached per segment_idx tensor | `cute/sliding_chunk_attention.py` | ~0.3 ms host per call saved | Only used by the cute SCA path |
| TF32 for the fp32 cross-entropy gradient GEMM | `train.py` already sets `allow_tf32` | 64 -> ~5 ms per step (harness only) | nothing |
| Profiling methodology (chrome-trace call-site attribution via `correlation` of cuda_runtime and cuda_driver events; `FWDBWD` window needs `torch.cuda.synchronize()` inside the `record_function`) | `handoff/scripts/profile/` | - | same scripts |

## 3. Items to port, in the order they paid off on B200

Each entry: what was built, what it bought, the H100-specific plan, how to validate. Re-order by the H100
baseline profile if it disagrees.

### 3.1 Sliding chunk attention (SCA): the biggest item on B200 (58% of the 0.9B step)

- **B200:** `xllm/modules/fused_ops/cute/sliding_chunk_attention.py`, `sca_backend='cute'`. Flash-style
  forward (q tile resident, 64-key K/V TMA ring, online softmax in 4 warps, P in TMEM as the A operand of PV,
  lse2 saved); backward = fused dK+dV kernel + dQ kernel (S + V <= 256) or three kernels (larger heads), delta
  via a row-dot kernel, V > 256 handled with 256-wide v slabs and cuBLAS fp32 dP GEMMs. Segment mask = per-row
  key range [kmin(row), row]; blocks fully outside are skipped. Numbers: 0.9B per layer fwd 0.83 / bwd 3.0 ms
  (swift ~12 / ~11), step 1151 -> 590 ms, peak memory 139 -> 40 GB; 4B bwd worst case 3.6 -> 1.8 ms/layer.
- **H100 plan:** first measure `sca_backend='xattn'` (forward and backward per layer, with `multi_segments=True`
  and the real segment structure) against the FA3 roofline: for 4B (hdim 128) FA3 does ~600-700 TFLOPS forward
  and ~450-550 backward on H100; for S = 256 heads (0.9B, 7.1B's qk dim) check whether xattn supports hdim 256
  at all and its speed. Only if xattn is missing, slow for these shapes, or wrong with segments, write an
  FA3-style CuTe DSL kernel: TMA producer warp, 2 consumer warpgroups (128 x 128 tiles for hdim 128), wgmma
  with S and O accumulators in registers, P converted to bf16 in registers as the A operand of PV, softmax
  ping-pong between the warpgroups, `setmaxnreg`. The backward on Hopper is the FA3 structure (per K/V tile:
  S^T, dP^T, dS in registers, dK/dV accumulators in registers, dQ via atomic adds to an fp32 buffer or a
  separate dQ kernel as done here). The design decisions that carry over: 128-row key tiles looping over the
  attending query blocks bounded by `query_limits`; the fused dK+dV pass (one S^T GEMM and one exp2 pass less
  than separate kernels); the dS scale applied in the epilogue; the masked / unmasked block bodies as separate
  code paths (ptxas if-converts an `if` around 160 selects, it then runs in every block - 10%); column data
  (lse, delta, kmin of a query block) prefetched one block ahead and staged through per-warp smem slots.
- **Validate:** `tests/test_cute_sliding_chunk_attention.py` (dense fp32 reference incl. segments; adapt the
  backend name), `handoff/scripts/sca/test_bwd.py <B,H,K,S,V,L,C,seg_p>`, then `profile/parity_sca.py` (swift
  vs new backend: losses, grad norms, step time, peak memory) on the full model.

### 3.2 TimestepDecayNorm (TSDN): CUB -> single-pass decoupled look-back scan

- **B200:** `xllm/modules/fused_ops/cute/timestep_decay_norm.py`, `timenorm_backend='cute'`. fwd 102 us / bwd
  156 us vs CUB 1488 / 3104 (B8 L8192 H2048 G32), 82-85% of HBM SOL; step 0.9B -8%, 7.1B -8.5%. Memory-bound.
- **H100 plan:** this kernel uses TMA, mbarriers, named barriers and the look-back protocol but no tensor
  cores and no TMEM, so it should port with small changes: (a) compile target sm_90a; (b) replace the
  Blackwell-only packed `fma_packed_f32x2` / `cute.arch.*_packed_f32x2` math with scalar ops (PTX `f32x2` ops
  are sm_100+); (c) re-tune T (tile length), the number of resolve warps and CTAs/SM for 132 SMs and 3.35 TB/s
  (the B200 study in HANDOFF.md §8 says what was tried); (d) the look-back latency matters more when there
  are few chains (B=1 L=65536). Target: >= 80% of the H100 HBM roofline (the CUB path moves ~10x the bytes).
- **Validate:** `tests/normalization/test_cute_timestep_decay_norm.py`, `profile/parity.py` (cub vs cute
  losses / grad norms).

### 3.3 AdaptiveWorkingMemory (AWM): torch reference -> phase-decomposed kernels

- **B200:** `cute/adaptive_working_memory.py` (+ `_bwd.py`, `_reference.py` = the phase decomposition as an
  executable spec), `awm_backend='cute'`. Identity kk (v - qk M) = kk v - (kk qk) M lets the A|B chunk GEMMs
  run in parallel over chunks, leaving a cheap S x V recurrence. fwd per call 0.9B 1376 -> 125 us, 4B 729 ->
  91, 7.1B 1259 -> 207; bwd 0.9B 99 -> 6 ms per step. ~10% of the step before, ~1-2% after.
- **H100 plan:** the algorithm (phases, workspaces saved for the backward, masks derived from segment_idx in
  kernel) carries over; the GEMM kernels (`AwmChunkGemm`, `AwmOutGemm`, `AwmDmoGemm`, `AwmChunkBwd`,
  `AwmRecurrenceUmma`) are tcgen05 and need wgmma versions: TMA + warp-specialized wgmma GEMM (the CuTe DSL
  Hopper GEMM example is the template), with the transform warps' work (P^T from k via lse, bf16 conversion,
  masks) done by the consumer warpgroups in registers. The recurrence (`AwmRecurrence`, warp-level bf16 mma
  with a cp.async ring) is Hopper-compatible as written except for packed f32x2 math. Keep the 7.1B
  (V = 2048) path on tf32 wgmma as it was on tcgen05. Before writing kernels, measure the torch backend on
  H100: on B200 AWM was ~10% of the step; if it is < 5% on H100 do it last.
- **Validate:** `tests/test_cute_adaptive_working_memory.py` (FP64 autograd reference, 19 tests), the harness
  with `AWM_BACKEND=cute` vs `torch` (losses equal).

### 3.4 SCA backward worst case (the §6b rewrite): what transfers

- **B200:** TMEM-resident A operands (K, V / Q, dO), 32-query blocks, two compute groups alternating blocks,
  raw mbarriers; 4B dK+dV 2546 -> 1071 us, dQ 1042 -> 710 us. The root causes (one CTA per SM, GEMMs with both
  operands in smem at ~55% of peak) are Blackwell-specific.
- **H100 plan:** nothing to port literally; on Hopper the equivalent is FA3's backward (register accumulators,
  2 consumer warpgroups, 128 x 128 tiles with hdim 128). The transferable lessons are the pipeline shape
  (dV_j, dK_{j-1}, dP_{j+1}, S_{j+2} issue order so the compute of block j has a full block of tensor work to
  hide behind), the measurement method (`SCA_DBG=1` no-math and `SCA_DBG=3` MMA-only variants of the kernel
  bound the compute side and the tensor side separately: do the same on H100 before optimizing anything), and
  the ptxas predication pitfall.

### 3.5 Host overhead of the forward (launch-bound models)

- **B200:** after the `create_on_gpu` fix the 4B forward was still launch-bound (1 GPU ~50 ms GPU idle per
  step): triton causal conv (108 launcher calls, ~17 ms), `F.linear` ~55 us per call, FSDP2 per-parameter python
  (~16 parameters per layer), the TSDN / AWM / SCA wrappers (0.2-0.3 ms per call). Not fixed.
- **H100 plan:** check first whether any H100 shape is launch-bound (`gaps.py`: compute-stream gaps in the first
  half of the step). H100 kernels run ~2x longer, so probably only gekko-4B at short L. If so: the conv
  launcher (3 triton launches per call: fuse the three conv calls per layer or cache the launcher), the
  kernel wrappers' `reshape` / dtype casts, and `fsdp_layers_per_group`.

### 3.6 gekko-7.1B specifics (V = 2048)

- **B200:** the SCA dV kernel recomputed S^T per 256-wide v slab (8x), 2.8 ms/layer. Now: `ScaPmat`
  materializes P (bf16, chunk-major [C x 2C] per (chunk, batch, head), zeros included) and dV is two cuBLAS
  bmm per chunk (1.47 ms/layer); since the GEMM path costs the same whatever the segment structure while the
  kernel path shrinks with it, `_use_pmat_dv` picks per call from the attended fraction (> 0.5). dP is a
  cuBLAS fp32 GEMM (1.2 ms); dK / dQ kernels read it (1.1 + 0.7 ms); dense model GEMMs are ~55% of the step.
- **H100 plan:** the materialize-P/dS + cuBLAS path is portable (any kernel that writes P, even a torch one,
  plus `ScaDsMat` and bmm) and on H100 the same crossover logic applies (re-measure the crossover: the kernel
  and GEMM costs scale differently). Do not fuse dS into the P kernel (three variants were slower on B200: the
  strided fp32 dP reads are latency bound). Check cuBLAS GEMM efficiency on H100 for the 7.1B shapes (dim
  4096, ffn 8192, V 2048 x 4 heads) vs the ~1 PFLOPS peak.

### 3.7 Elementwise and norm fusions (HANDOFF §6d): second-largest item after SCA on 0.9B / 4B

- **B200:** the per-op attribution (`profile/opattr.py` on a `RECORD_SHAPES=1` trace) showed 55% of the 0.9B
  kernel time in elementwise / norm / copy kernels. Seven fused kernels (rope, q/k prep, attention gate,
  swiglu, cross-entropy chunk, layernorm) and a fix of the extension's norm weight-gradient kernels took the
  0.9B step 469 -> 335 ms, 4B 390 -> 325, 7.1B 924 -> 826; all are memory-bound kernels at 4-6.5 TB/s on B200.
- **H100 plan:** all of them are plain memory kernels and port as they are (lower the per-SM thread counts
  only if occupancy limits show up); the gains scale with H100's 3.35 TB/s vs 8 TB/s, so the share of the step
  they save is similar or larger. Do the csrc gamma fix first (it is a rebuild, no port), then rope and q/k
  prep (largest), then the gate, swiglu, CE, layernorm. Keep each behind its env switch and validate with
  `profile/parity_env.py` (loss to 1e-4, grad norms to 1e-3) and the unit tests in `tests/test_cute_*.py`.
- **Not done:** the causal conv1d (Triton, ~2.3 TB/s, 20 / 34 / 18 ms per step on 0.9B / 7.1B / 4B); two CuTe
  drafts in `handoff/scripts/conv/` are correct but slower (latency bound). On H100 the Triton kernel's share
  grows; a TMA-tile + time-parallel design is the next attempt.

## 4. How to work (what worked on B200)

- One item per commit, each with: microbenchmark before/after at the real shapes (per-call GPU time via
  torch.profiler, like `prof_bwd.py`), the test file, the harness step time and loss / grad-norm parity, and
  the numbers in the commit message. Push after each.
- ncu is the tool for kernels: `--set full --import-source yes --page source --print-source cuda,sass --csv`
  with `CUTE_DSL_LINEINFO=1`, aggregated by `common/src_an.py` (per-line instruction counts, stall reasons,
  and whether code runs predicated). Check `sm__pipe_tensor_cycles_active` (tensor pipe busy) and
  `sm__inst_executed_pipe_xu` (MUFU: exp2 bound shapes) first.
- Build measurement variants into the kernel behind an env var (`SCA_DBG` style): no-math, MMA-only,
  no-TMA. They turn "it is slow" into "the tensor side alone takes X, the compute side adds Y".
- CuTe DSL pitfalls that are not Blackwell-specific (HANDOFF.md §8, `handoff/memory/*.md`): no closures
  called inside staged for/if (pass arguments to a `@cute.jit` helper instead); `cutlass.range_constexpr`
  only inside jit/kernel functions; `tm.set(ACCUMULATE, dynamic)` leaks staged state out of helpers; TMA
  `cute.copy` must be issued by a full warp; a producer stage that nobody consumes hangs; "empty" barriers
  with no first producer start at parity 1; TVM-FFI fake tensors are int32-sized (2-D fakes for > 2^31
  elements); results of look-back scans and `red.add` reductions are not bitwise deterministic.
- Memory in this directory (`handoff/memory/`) is the B200 session's auto-memory; copy what is still true
  into the H100 session's memory directory and correct the arch-specific parts.
