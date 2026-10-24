# CuTe DSL kernels for xllm: handoff

This handoff covers four tasks:

- `TimestepDecayNorm` (TSDN): forward and backward are done.
- `AdaptiveWorkingMemory` (AWM): forward and backward are done (section 6 has what is left).
- Sliding chunk attention (SCA): forward and backward are done (section 6b), after a full-step profile showed the
  swift SCA at 58% of the 0.9B step on B200.
- 8-GPU FSDP2 overlap investigation (section 6c): the collectives were already hidden; it found a default-device
  bug costing ~10% of the CPU time of every step, added `fsdp_layers_per_group`, and quantified data-dependent
  straggling.

Both are written in the CuTe Python DSL (nvidia-cutlass-dsl 4.8.1) and target B200 (sm_100a).

The original request came from Max: replace the CUB/torch implementations with FA-level-performance CuTe kernels. Use ncu to find bottlenecks. Push finished commits to `origin` (github.com/Johnsonms/xllm).

`H100_PLAN.md` next to this file lists every optimization with its effect and how to redo it on H100 (Hopper).

## 1. Branches (all pushed to origin)

| Branch | Content |
|---|---|
| `cute-timestep-decay-norm` | TSDN fwd+bwd kernels, tests, and the `timenorm_backend='cute'` wiring (10 commits on `main`). |
| `cute-adaptive-working-memory` | The TSDN branch plus the AWM forward and backward kernels and the SCA kernels, tests and the `awm_backend='cute'` / `sca_backend='cute'` wiring. **This is the working branch.** |
| `cute-kernels-handoff` | `cute-adaptive-working-memory` plus this `handoff/` directory (dev scripts, notes, venv freeze). Do not merge it; it is for reference only. |

Commit convention (no git identity is configured on the machines):

```
git -c user.name="Johnson" -c user.email="johnson@together.ai" commit ...
# message ends with:
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```

## 2. Environment setup on a new node

The steps below assume 8× B200, driver with CUDA 13, and system python 3.10. (On slinky-0 the system CUDA 13.1 nvcc and ncu in /usr/local/cuda/bin work directly: no pip toolchain or ncu download needed; `xllm_extension` built with `CUDA_HOME=/usr/local/cuda`.) The repo lives at `~/xllm` and the venv at `~/xllm-venv`. Many scratch scripts hardcode `/home/johnson/xllm`; clone there, or symlink it.

```bash
git clone https://github.com/Johnsonms/xllm.git ~/xllm && cd ~/xllm && git checkout cute-adaptive-working-memory
python3 -m venv ~/xllm-venv
~/xllm-venv/bin/pip install --upgrade pip
~/xllm-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130     # 2.14.1+cu130
~/xllm-venv/bin/pip install nvidia-cutlass-dsl==4.8.1 apache-tvm-ffi einops pytest ninja
# for running the Gekko model / the fused block (installed with --no-deps):
~/xllm-venv/bin/pip install --no-deps fla-core==0.5.2 flash-linear-attention==0.5.2
~/xllm-venv/bin/pip install typeguard fire sentencepiece tiktoken==0.8.0 blobfile submitit nvidia-ml-py
```

The exact package list is in `handoff/venv-freeze.txt`.

### CUDA toolchain for building extensions

There is no system nvcc. Use pip nvcc, pinned to **13.0.\*** to match torch's cu13 headers. The nvcc, cccl, nvvm and crt packages must all be on 13.0.

```bash
~/xllm-venv/bin/pip install "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-cccl==13.0.*"   # + nvvm/crt 13.0.* if pulled separately
N=~/xllm-venv/lib/python3.10/site-packages/nvidia/cu13
mkdir -p ~/scratch/cubext/libs && ln -sf $N/lib/libcudart.so.13 ~/scratch/cubext/libs/libcudart.so
```

### xllm_extension (needed only for the full Gekko model harness)

The build takes about 4 minutes:

```bash
S=~/scratch; N=~/xllm-venv/lib/python3.10/site-packages/nvidia/cu13
cd ~/xllm && CUDA_HOME=$N PATH=$N/bin:~/xllm-venv/bin:$PATH CPATH=$N/include/cccl:$N/include \
  LIBRARY_PATH=$S/cubext/libs TORCH_CUDA_ARCH_LIST=10.0 MAX_JOBS=32 \
  ~/xllm-venv/bin/python setup.py build_ext --build-lib $S/xext --build-temp $S/xext_tmp
```

`run_gekko.py` puts `<script dir>/xext` on `sys.path`. Copy it next to the built `xext/`, or edit the path.

### ncu (not installed system-wide; no sudo needed)

```bash
mkdir -p ~/scratch/tools && cd ~/scratch/tools && apt-get download nsight-compute-2026.3.1 && dpkg -x nsight-compute-2026.3.1*.deb ncu
export NCU=~/scratch/tools/ncu/opt/nvidia/nsight-compute/2026.3.1/ncu
```

Profiling works without root.

To get register counts with `ptxas -v`, the 13.0 ptxas cannot read the DSL's PTX 9.4. Download the `nvidia-cuda-nvcc==13.4.*` wheel, unzip it somewhere (do not install it), and use its ptxas.

### Gekko harness networking

On a single node, run with `NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1`. The harness stubs `flash_attn_2_cuda`.

### Claude Code memory

Copy `handoff/memory/*.md` into `~/.claude/projects/<project-dir>/memory/` and add these two lines to `MEMORY.md`:

```
- [xllm CuTe TSDN kernel](xllm-cute-tsdn.md) — CuTe DSL TimestepDecayNorm task: venv, CUB build gotchas, baseline perf
- [xllm CuTe AWM kernel](xllm-cute-awm.md) — AWM task: real Gekko shapes, 1-GPU harness, env setup
```

They hold the detailed lessons: DSL pitfalls, perf studies, and failed experiments.

## 3. Verify the node

```bash
cd ~/xllm
~/xllm-venv/bin/python -m pytest tests/test_cute_adaptive_working_memory.py -q          # 20 pass
~/xllm-venv/bin/python -m pytest tests/normalization/test_cute_timestep_decay_norm.py -q # 27 pass
cd handoff/scripts/awm
~/xllm-venv/bin/python bench_awm_ev.py 2,8,4,256,256,16384,4096 1,32,8,128,128,16384,4096 1,4,4,256,2048,16384,4096
#   expect about 127 / 118 / 207 us per call (back-to-back events; 4B is host-bound)
NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 AWM_BACKEND=cute ~/xllm-venv/bin/python run_gekko.py gekko-0.9B 2 16384
```

The first call of each shape JIT-compiles, which takes tens of seconds.

## 4. TSDN: done

Files:

- Kernel: `xllm/modules/fused_ops/cute/timestep_decay_norm.py`.
- Tests: `tests/normalization/test_cute_timestep_decay_norm.py`.
- Wiring: `timenorm_backend='cute'` in `xllm/config.py`, `TimestepDecayNorm`, and the fused Gekko blocks.

Performance (bf16, B=8, L=8192, H=2048, G=32):

| Pass | CuTe | CUB | % of HBM SOL |
|---|---|---|---|
| Forward | 102 µs | 1488 µs | about 85% |
| Backward | 156 µs | 3104 µs | about 82% |

The design is a single pass with a decoupled look-back scan. Results are not bitwise deterministic, because of the look-back order and dgamma accumulated with `red.add`.

There is little headroom left. Section 8 lists the studies.

## 5. AWM forward: done

Files:

- `xllm/modules/fused_ops/cute/adaptive_working_memory.py`, about 1300 lines. The docstring has the kernel list and a performance table.
- `tests/test_cute_adaptive_working_memory.py`: an FP64 port of the reference and 9 forward cases, including the 0.9B and 7.1B shapes.
- `xllm/modules/fused_ops/cute/adaptive_working_memory_reference.py`: the **executable spec of the phase-decomposed backward**, `awm_bwd_phased`. It matches autograd to 1e-12 in FP64, with segments and memory resets.

### Wiring

- `ModelConf.awm_backend` takes `'torch'` or `'cute'`. The path is `xllm/models/gekko.py` → `GatedDeltaAttention` → `AdaptiveWorkingMemory(backend=)`, and the fused block `xllm/models/fused_blocks/gekko/gekko.py`.
- `xllm/modules/fused_ops/adaptive_working_memory.py::adaptive_working_memory_fwd(..., backend)` dispatches to CuTe when `_cute_supported` holds: bf16, `ortho=False`, compute capability 10.x, S∈{128,256}, V%64==0, and chunk%128==0. Otherwise it falls back to torch.
- **The backward is still the torch reference**: `_accum_fwd` recompute plus `_bwd`.

### Algorithm

The decomposition rests on the identity kk(v − qk·M) = kk·v − (kk·qk)·M. It runs in three phases:

1. **`AwmKeyLsePart`**: per-chunk logsumexp of the keys over time, split by rows, with masks from `segment_idx` computed in the kernel. **`AwmChunkGemm`** (tcgen05) then computes [A|B] = softmax_t(kᵀ)·[qk|v] for every chunk in parallel. Its design:
   - MN-major operands.
   - Transform warps compute P = exp2(k·log2e − lse2).
   - Separate k, P and B smem rings.
   - The epilogue uses TMA reduce-add.
2. **Recurrence** M ← (1−r)M + r(B − A·M) over the chunks, which is sequential:
   - **`AwmAToBf16` + `AwmRecurrence`** (warp bf16 mma with a cp.async ring) for V<1024.
   - **`AwmRecurrenceUmma`** (tcgen05 tf32, with the state Mᵀ held in smem) for V≥1024.
3. **`AwmOutGemm`**: out = q·M_c. It is persistent, keeps the M tile resident, and has 2 epilogue groups. It also writes `out_mask`.

### Performance

Forward GPU time per call (bf16, L=16384, with segments):

| Model | Shape | Torch reference | CuTe |
|---|---|---|---|
| gekko-0.9B | B2 H8 K4 S256 V256 | 1376 µs | 125 µs |
| gekko-4B | B1 H32 K8 S128 V128 | 729 µs | 91 µs |
| gekko-7.1B | B1 H4 K4 S256 V2048 | 1259 µs | 207 µs |

On 0.9B the time splits as out GEMM ~46 µs, chunk GEMM ~38 µs, recurrence ~26 µs and lse ~19 µs.

End-to-end training on 1 GPU:

| Model | AWM forward per step | Step time | Losses |
|---|---|---|---|
| 0.9B | 20.7 → 2.5 ms | — | identical |
| 7.1B | 40.1 → 6.8 ms | 1263 → 1222 ms | match to ≤4e-6 |

Real training shapes are listed in `handoff/memory/xllm-cute-awm.md`:

- C=4096 and `ortho=False`.
- `segment_idx` is passed.
- Memory is None on CP rank 0.

### Open forward issue

`AwmRecurrenceUmma` on 7.1B: every CTA exits at about 40 µs, but the loops end at 22–32 µs. About 10–15 µs of tail time is unexplained: it is not the Mᵀ writes, and the mem_out store and TMEM free are suspects. The globaltimer probe for this is `handoff/scripts/awm/recprobe/`.

## 6. AWM backward: done (2026-10-23, node slinky-0)

Files:

- `xllm/modules/fused_ops/cute/adaptive_working_memory_bwd.py`: the backward kernels and host side. Its docstring
  lists the kernels and the performance table.
- The forward (`adaptive_working_memory.py`) gained `return_workspace=True` (the tensors in `WORKSPACE_KEYS`:
  lse partials, [A | B], bf16 A, M sequence, D, r, -M, dead flags, M^T; ~29 MB per layer on 0.9B, ~90 MB on
  7.1B) and the recurrence kernels a `save` mode; the host side is split into `_phases12` / `_out_gemm`.
- Wiring: `AdaptiveWorkingMemoryFunc` saves the workspace and calls `awm_bwd_prepare` / `awm_bwd_finish`; the
  fused Gekko block does the same (it recomputes the AWM output between the two halves). `awm_backend='cute'`
  falls back to the torch recompute + backward when an input memory / previous chunk is given (context-parallel
  ranks > 0) or V is not a multiple of S.
- Tests: `tests/test_cute_adaptive_working_memory.py` (20: forward, phased-backward spec, backward vs FP64
  autograd on the Gekko shapes, saved-workspace path).

Algorithm: the mirror of the forward phases (spec in `adaptive_working_memory_reference.py::awm_bwd_phased`):
3' dq = dOut M_c^T (the forward output GEMM with S and V swapped, streamed memory tile when it does not fit) and
dMo = sum_h q_c^T dOut_c (split-K GEMM, masked time tiles skipped); 2' reverse recurrence G <- (1 - r) G -
A^T (rG) + dMo with dB = rG written and dr partials, dA = dB (-M)^T as a batched matmul afterwards, then delta =
rowdot(dA, A) + rowdot(dB, B) (= sum_t P dP) and the lse chain; 1' per 128 time rows: dP^T = qk dA^T + v dB^T,
P^T = exp(k - lse), dk = P^T (dP^T - delta + dlse), dqk = P^T dA, dv = P^T dB.

GPU time per call (bf16, L = 16384, with segments, saved workspaces):

| Model | CuTe backward | Torch recompute + backward |
|---|---|---|
| gekko-0.9B | 236 us (chunk 88, dq 49, dMo 48, recurrence 30) | ~3.5 ms |
| gekko-4B | 142 us | ~2.5 ms |
| gekko-7.1B | 400 us (chunk 181, dq 72, dMo 72, recurrence 30) | ~3.1 ms |

Training step on 1 GPU (`run_gekko.py`): AWM backward 99 -> 6 ms per step (0.9B, B2 L16K, step 1409 -> 1262 ms),
98 -> 11 ms (7.1B, B1, step 1274 -> 1149 ms); gradient norms match the torch backend.

### Where the time goes and what was tried

- `AwmChunkBwd` (persistent, 2 epilogue warp groups, TMA-store epilogues, 32-row operand blocks in 4-stage
  rings): per 128-row tile ~15 us = dP^T GEMM phase ~7 us (bound by ring bytes / TMA round trip: the dA | dB
  operand, 256 KB per tile, is re-streamed by every tile) and the three epilogues ~7 us, which cannot overlap the
  next tile's GEMM with only two TMEM accumulators. Tried: separate dP^T / dqk-dv pipelines (the next dP^T starts
  after the dk epilogue, but dv then waits for the dqk epilogue: slower, 95 us); dqk / dv in two TMEM halves
  placed with pointer offsets (wrong results, cause not found - note that the SCA kernels and the official FMHA
  example use `tmem_ptr + offset` / `recast_ptr` for every TMEM region without trouble, so the earlier advice
  to avoid pointer offsets was a misdiagnosis); issuing the last raw key blocks before the GEMM loads (slower).
  Next: a 2-CTA (cta_group::2) dP^T GEMM halves the B bytes per CTA; 7.1B (K = 2304) would gain most.
- `AwmDmoGemm` is at ~56% of HBM SOL; `AwmOutGemm` (dq) is the forward kernel.
- `AwmRecurrenceBwd`: ~10 us per update; the dA GEMM was moved out of the chain (batched matmul).
- Feature gap: context-parallel ranks > 0 (input memory and previous chunk) use the torch backward. The kernels
  have the hooks (`first`, `has_prev`, `has_dmem`, `_seg_anchor`) but the chunk kernel reads the update chunk
  from the current sequence only and the previous-chunk gradients are not produced.

## 6b. Step profile and sliding chunk attention (SCA): done (2026-10-23/24, slinky-0)

### Where a Gekko step went (before SCA)

Full-step profiles (`handoff/scripts/profile/prof_step.py`, `prof_stack2.py` + `attrib.py`: per-kernel times
attributed to python call sites through the chrome trace; `fsdp_step.py`: 8-GPU FSDP2 steps). 1 GPU, L = 16384,
cub timestep norm, cute AWM:

| | gekko-0.9B B2 (1258 ms/step) | gekko-7.1B B1 (1150 ms/step) |
|---|---|---|
| swift sliding chunk attention (fwd + bwd) | 671 ms (58%) | 265 ms (23%) |
| dense GEMMs | ~260 ms | ~580 ms (54%) |
| timestep norm (CUB) | ~120 ms | ~130 ms |
| cross-entropy grad GEMM (fp32 SIMT in the harness; TF32 in train.py) | 64 ms | 67 ms |
| rope (cat + complex), norms | ~70, ~65 ms | ~18, ~45 ms |
| AWM (cute) | ~10 ms | ~23 ms |

The swift SCA is cuBLAS batched GEMMs plus a materialized bf16 attention matrix (~105 GB of the 139 GB peak on
0.9B) and ~300 ms of contiguous copies. xattn (the flash-style SCA used on H200) does not support SM100.
`timenorm_backend='cute'` alone saves 8.5% (loss / grad parity verified in the model). On one node FSDP2
communication was fully hidden at 8 GPUs.

### SCA kernels

- `xllm/modules/fused_ops/cute/sliding_chunk_attention.py`: `ScaFwd`, `ScaBwdDv`, `ScaBwdDk`, `ScaBwdDq`,
  `RowDot` (delta), `ChunkMajorCopy`; host `sca_fwd` / `sca_bwd`, helpers `segment_starts` (kmin[b, t]) and
  `query_limits` (qmax[b, k]), a dense `sca_reference`. Wired as `sca_backend='cute'` in
  `xllm/modules/fused_ops/attention/sca.py` (`cute_sca_supported`: qk dim <= 256, v dim multiple of 64 up to 256
  or multiple of 256, chunk multiple of 128, no previous chunk, no dropout; otherwise swift with a one-time
  warning). Tests: `tests/test_cute_sliding_chunk_attention.py` (19).
- Design: flash-attention style. Forward per (b, h, chunk, 128-row q tile[, 256-wide v slab]): q tile resident in
  smem, 64-key K / V blocks through 2-stage TMA rings, S in TMEM (2 x 64 columns), P bf16 stored into TMEM as the
  A operand of the PV MMA (as the CuTeDSL fmha example does), O 256 columns; 4 softmax warps (thread = row), lazy
  O rescale (threshold 8 in log2 units, warp-uniform via `vote_any_sync`). The mask is the per-row key range
  [segment start, row] (segment ids are nondecreasing); blocks outside it are skipped, so documents shorter than
  the window cut the work. Saved for the backward: y and lse2 (log2 domain, scale folded).
- Backward: `ScaBwdDv` per key tile (loops over the heads of the kv group x the 64-query blocks up to
  `query_limits`; P^T into TMEM, dV += P^T dO), `ScaBwdDk` per key tile (V tile in TMEM as the A operand of
  dP^T = V dO^T; the Q block's bytes serve as K-major B of S^T and MN-major B of dK += dS^T Q, the two SW128
  layouts are identical), `ScaBwdDq` per q tile (mirror, dO in TMEM). For V > 256 (7.1B) the forward / dV run
  per 256-wide v slab (each slab CTA recomputes S) and dK / dQ read dP from fp32 cuBLAS GEMMs
  (`torch.bmm(..., out_dtype=float32)` on chunk-major copies). A dS-materialized variant (write dS^T, dq by
  cuBLAS) was slower (zero fill + transposed copies) and needed 4.3 GB.

Per layer, B200, L = 16384 (bf16, segments every ~3000 tokens):

| Model | cute fwd | cute bwd | swift fwd | swift bwd | step 1 GPU | peak memory |
|---|---|---|---|---|---|---|
| gekko-0.9B B2 | 0.83 ms | 3.0 ms | ~12 ms | ~11 ms | 1151 -> 590 ms | 139 -> 40 GB |
| gekko-4B B1 L8192 | | | | | 967 -> 455 ms | 142 -> 37 GB |
| gekko-7.1B B1 | 0.77 ms | 3.4 ms | ~3.8 ms | ~4.5 ms | 1050 -> 968 ms | 98 -> 81 GB |

Losses and gradient norms match swift to bf16 noise (`handoff/scripts/profile/parity_sca.py`). 8 GPUs FSDP2
(`fsdp_step.py`, all cute backends): 0.9B fwd+bwd 605 ms (communication hidden), 7.1B 1043 ms and 4B (L8192)
565 ms against 968 / 455 ms on 1 GPU: with the compute this much shorter, the FSDP all-gathers / reduce-scatters
are no longer hidden on those two (nccl 481 ms on 7.1B; busy(non-nccl) 991 of a 1039 ms span, and the step is
longer than on 1 GPU). That is the next thing to look at for multi-GPU training (reshard_after_forward=False,
prefetching, or bucketing).

### Backward rewrite for S + V <= 256 (2026-10-24, after §6c; commit 459db31)

Asked for: the worst case of the backward (batches without segment boundaries, the straggler of §6c). On the
4B shape (H32 K8 S128 V128, B1 L8192) it took 3.6 ms per layer with the tensor pipe 30-40% busy. ncu plus an
MMA-only variant of the kernel (`SCA_DBG=3`) showed two causes: 512 TMEM columns per CTA leave one CTA per SM,
so every handshake latency between the MMA warp and the compute warps is exposed; and the N = 64 GEMMs with
both operands in shared memory run at ~55% of the tensor-core rate (operand bandwidth, ~192 B/clk needed).

- `ScaBwdDkDv`: dK and dV of a key tile in one pass (one S^T GEMM and one exp2 pass less). K and V are stored
  in tensor memory (bf16, by the compute warps) as the A operands of S^T = K Q^T and dP^T = V dO^T; P^T is
  written over the S^T columns as the A operand of dV; S^T / dP^T are double-buffered 32-query blocks
  (TMEM: K 64 | V 64 | S^T x2 | dP^T x2 | dV 128 | dK 128 = 512). Two groups of 4 compute warps alternate
  blocks; the MMA warp issues dV_j, dK_{j-1}, dP^T_{j+1}, S^T_{j+2} per block over raw mbarriers (an S^T / P^T
  stage is freed by the completion of its dV MMA through a tcgen05 commit on the stage's empty barrier, a dP^T
  stage by the compute warps, a dS^T buffer by its dK MMA). Q ring 4 stages, dO ring 3.
- `ScaBwdDqT`: same idea for dQ (Q and dO in TMEM, 64-key blocks, S / dP double-buffered, 2 dS buffers).
- The per-block body is a `@cute.jit` helper with explicit arguments, instantiated for masked and unmasked
  blocks: an `if` around the mask selects alone is if-converted by ptxas (every block paid ~160 predicated
  instructions, 10%). Column data (lse2, delta, kmin of the block) is prefetched one block ahead and staged
  through per-warp shared-memory slots (no group barrier); the dS scale moves to the epilogue; dP is loaded
  after the P store (96 registers, no spills; 320 threads cap registers at 168).
- Numbers (4B, per layer, worst case): dK + dV 2546 -> 1071 us, dQ 1042 -> 710 us, backward 3627 -> 1824 us
  (MMA-only floors 753 / 562 us: the remaining gap is compute-warp latency and the grid tail, 3.5 waves).
  Model step, 1 GPU, batches without eos: 474 -> 424 ms mean, worst step 520 -> 458; 8 GPUs 494 -> 480
  (`fsdp_layers_per_group=1`), 485 -> 428 (4). Losses / grad norms unchanged; 19 SCA tests pass.
- Scope: S + V <= 256 only (TMEM budget 1.5 (S + V) + 4 QB <= 512); 0.9B / 7.1B (S = V = 256) keep the
  previous kernels. The same A-in-TMEM structure would apply to the forward (Q tile in TMEM) and, with a
  materialized bf16 P^T + cuBLAS dV, to the 7.1B dV (2.8 ms/layer today).
- Scripts: `sca/ncu_sca.py <shape>` (one profiled backward for ncu), `common/src_an.py <csv> <top> <source>`
  for the per-line exec counts / stalls from `ncu --set full --import-source yes --page source --print-source
  cuda,sass --csv` with `CUTE_DSL_LINEINFO=1` (it shows `@P` predication), `sca/layout_check{2,3}.py`.

### Forward with the q tile in tensor memory (2026-10-24, commit 8a8e57a)

`ScaFwdT` applies the same move to the forward: q in TMEM as the A operand of S = Q K^T, P over the S columns,
raw mbarriers (an S/P stage is freed by the completion of its PV MMA). gekko-4B needs 256 TMEM columns with
32-key blocks, so two CTAs share an SM; this only works when the TMEM allocation permit is relinquished right
after the allocation (`tmem.relinquish_alloc_permit()` before the main loop), otherwise the second CTA waits
for the first to exit and the kernel is slower than before (878 us). S = 256 shapes use 64-key blocks (the
N = 32 variant costs 2x the MMA instructions: 674 vs 540 us on 0.9B). Per call: 4B 668 -> 550 us
(segmented 402 -> 340 / 583 -> 473), 0.9B 558 -> 540, 7.1B unchanged (2.07 ms: the 8 v-slab CTAs each
recompute S, which is as many flops as their PV; a materialized bf16 P^T + cuBLAS PV would be ~1.8 ms).
Floors (4B): MMA-only 358 us, no-math 393 us; the full kernel is exp2 bound (1.07 G exp2 per layer = 238 us
of MUFU). Step times move by <= 1% (the forward is ~5% of the step). `SCA_FWD_TMEM=0` selects the old kernel.

### 7.1B dV from a materialized P (2026-10-24, commit b3e435f)

`ScaPmat` (q tile in TMEM, 64-key blocks, two CTAs per SM) writes P = exp2(S scale2 - lse2) bf16 chunk-major
as [C queries x 2C keys] per (chunk, batch, head), including the zeros of the unattended prefix and the causal
suffix of each row (no memset), and dV of the keys of chunk c is `bmm(P_c[:, C:]^T, dO_c)` plus
`baddbmm_(P_{c+1}[:, :C]^T, dO_{c+1})` on the chunk-major dO copy, written back by `ChunkMajorUncopy`.
Per layer (7.1B, no segment boundaries): dV 2.86 -> 1.47 ms, backward 6.13 -> 4.92 ms. The GEMM path's
cost is fixed (dense [C x C] blocks) while the v-slab kernel's shrinks with the segment structure
(1.1 ms at 7 segments per 16K tokens, 0.58 at 19), crossover at an attended fraction of ~0.5, so
`_use_pmat_dv` computes that fraction from qmax (one `.item()` per layer backward) and picks the path;
`SCA_DV_PMAT=0/1` forces one. Training batches with a few eos per sequence take the kernel path, so the 7.1B
step is unchanged (939 ms); batches without boundaries (the stragglers of §6c) get the GEMM path.

dK / dQ on the same path (commit 88b246b): dP is computed as fp32 [queries x keys] GEMMs (dO_c V_c^T,
dO_c V_{c-1}^T), `ScaDsMat` writes dS = P (dP - delta) scale bf16 in the layout of P, and dQ_c =
dS_c[:, C:] K_c + dS_c[:, :C] K_{c-1}, dK_c = dS_c[:, C:]^T Q_c + dS_{c+1}[:, :C]^T Q_{c+1} are bmm /
baddbmm_ on chunk-major copies of K and Q. Per layer without boundaries: dK + dQ 1.8 -> 1.0 ms, backward
6.13 -> 4.16 ms (kernel path 2.52 ms at 19 segments per 16K, unchanged). Model losses / grad norms with the
path forced match the kernel path to bf16 noise. Fusing the dS computation into `ScaPmat` was tried three
ways (register reads, early-issued reads, cp.async staging one block ahead) and was always slower
(1.2-1.5 ms vs 0.46 + 0.52): the per-thread 256-byte strided fp32 dP reads are latency bound and the
2-CTA/SM budget (registers <= 168, smem <= 114 KB) leaves no room for deeper prefetch. The remaining cost of
the path is dominated by the dense GEMMs over half-empty causal blocks (dP 1.2 ms, dV 1.06 ms): splitting the
[C x C] causal blocks into 2 x 2 sub-blocks and skipping the upper one would save ~25% of them.

### Open items

- The S = 128 shape (4B) is MUFU bound (exp2); FA4-style exp2 emulation on the FMA units for part of the
  elements would help. For S = 256 the kernels are near the MMA rate (~1.4 PFLOPS effective incl. masked waste).
- 7.1B backward: dP GEMM 1.2 ms + dV 1.1 ms per layer dominate; a fused dP (streaming V and dO slabs) would
  exceed the L2 bandwidth, so it stays a GEMM.
- Context-parallel ranks > 0 (previous chunk) fall back to swift, as for AWM.
- Remaining step time (0.9B, 590 ms): dense GEMMs ~250, fused-block elementwise ~90, cross-entropy ~85 (TF32 in
  train.py makes it ~5), swiglu ~90, rope ~70, norms ~65, SCA ~115, AWM ~10.
- (Done in §6d: rope, the q/k softmax chain, the norm weight gradients, the attention gate, swiglu, the
  cross-entropy chunk ops, layernorm, and TMEM-A kernels for the S = V = 256 backward.)

## 6c. FSDP2 overlap on 8 GPUs: done (2026-10-24, slinky-0)

Question (after §6b): with the shorter cute kernels, 8-GPU FSDP2 steps looked slower than 1-GPU steps (7.1B 1043
vs 968 ms, 4B 565 vs 455). Method: `profile/fsdp_trace.py` (torchrun, all cute backends, chrome trace of one
step on rank 0 with a `FWDBWD` annotation that includes a `torch.cuda.synchronize()`, so the CPU annotation is
the fwd+bwd window), `gaps.py` (compute-stream idle time inside that window and whether NCCL was running),
`gapattr.py` / `cpuself.py` (GPU-idle time and CPU self time by python frame, `WITH_STACK=1`), `kcmp.py`.

### Findings

1. **Communication is overlapped.** Compute-stream gaps under NCCL are <= 15 ms per step on all three presets
   (the first all-gather of the forward and the last reduce-scatter of the backward are the only exposed ones).
   `fp32_reduce_scatter=False` and `reshard_after_forward=False` change nothing (7.1B 1021 -> 1017 / 1019 ms).
   NCCL kernel durations in the profile (~450 ms on 7.1B) are mostly waiting for peers, not bandwidth.
2. **The 8-GPU "penalty" is straggling from data-dependent compute.** The cute SCA skips key blocks across
   segment boundaries, so a rank's step time depends on its batch: with one seed, consecutive 4B steps take 432
   to 520 ms (SCA kernels 79 vs 155 ms; a sequence without eos is the worst case), reproducibly and independent
   of CPU/NUMA pinning. The 8-GPU step is the max over ranks: with identical batches on all ranks 7.1B = 960 ms
   (1 GPU 942) and 4B = 449 (1 GPU 444). Not an FSDP problem; the levers are a faster worst-case SCA backward
   (S = 128: ~0.57 PFLOPS vs ~1.2 in the forward) or work-balanced batches.
3. **Bug (fixed, commit 671884b):** `create_on_gpu()` restored the default device with
   `set_default_device("cpu")`, which installs a `DeviceContext` TorchFunctionMode (python `__torch_function__`
   on every torch op) for the rest of the process: 24k calls per 4B forward. Restoring the raw setting (`None`)
   removes it. 1 GPU: 4B 554 -> 444 ms, 0.9B 590 -> 530; 8 GPUs: 4B 614 -> 504, 0.9B 600 -> 591. train.py uses
   `build_model` -> `create_on_gpu`, so production runs had it too.
4. **gekko-4B's forward is launch-bound** even after the fix (1 GPU: ~50 ms GPU idle per step; 8 GPUs 60-80 ms,
   the FSDP2 hooks add per-parameter python work, ~16 parameters per layer). Remaining host costs per step:
   triton causal conv (108 calls, ~17 ms of launcher time), `F.linear` ~55 us per call (296 calls), FSDP2
   `_get_param_all_gather_inputs` / copy-out / `alloc_storage`, the TSDN / AWM / SCA wrappers (0.2-0.3 ms per
   call). `ModelConf.fsdp_layers_per_group` (commit 1f9d3c0) groups consecutive GekkoBlocks into one
   `fully_shard` parameter group (layer 0 alone, so the exposed collectives stay small): 4B 8 GPUs 504 -> 485 ms
   at 4 (same batches: 449 -> 430), +1 GB peak memory, loss / grad norm parity to bf16 noise; 6 and 9 are not
   better; 0.9B and 7.1B gain nothing (not launch-bound). The SCA `segment_starts` / `query_limits` cache their
   result per segment_idx tensor (all layers of a step share it).

### Numbers (fwd+bwd ms, B200, torchrun, cute TSDN/AWM/SCA, per-rank random batches unless noted)

| preset | 1 GPU before | 1 GPU after fix | 8 GPUs before | 8 GPUs after fix | 8 GPUs grp=4 | 8 GPUs same batches |
|---|---|---|---|---|---|---|
| 0.9B B2 L16384 | 590 | 530 | 600 | 591 | 585 | - |
| 4B B1 L8192 | 554 | 444 | 614 | 504 | 485 | 449 (grp=1) / 430 (grp=4) |
| 7.1B B1 L16384 | 946 | 942 | 1021 | 1031 | 1035 | 960 |

### Open items

- Launch-bound 4B forward: CUDA graphs are out (FSDP2 reallocates the unsharded parameters every step); the
  per-call host costs above are the list to shrink (fewer triton launches in the conv, fewer `reshape` / dtype
  casts in the kernel wrappers, fusing the three conv calls per layer).
- Straggling: a data loader that balances attention work (segment structure) across ranks, or a faster
  worst-case SCA backward.
- Per-rank step-time telemetry in train.py would make straggling visible in production.

## 6d. Second pass (2026-10-24, autonomous 12-hour run): fusions, norm kernels, SCA backward for S = V = 256

Starting point (1-GPU fwd+bwd, cute TSDN / AWM / SCA): 0.9B B2 L16K 527 ms, 4B B1 L8K 424, 7.1B B1 L16K 939.
End: 0.9B 319, 4B ~323 (launch-bound; time it alone, CPU load from other jobs moves it by 10%), 7.1B ~820;
8 GPUs 7.1B 884 (was 1017), 0.9B 388. Every item: commit on `cute-adaptive-working-memory`, unit test under
`tests/test_cute_*.py`, loss / grad-norm parity on the harness via `profile/parity_env.py` (same batches, env
var toggles the path in one process), step time via `profile/run1.sh`. All new paths have an env kill switch.

### What the profile said

`profile/opattr.py <trace> <kernel regex>` (needs a trace recorded with `WITH_STACK=1 RECORD_SHAPES=1
run1.sh ...`) attributes every kernel to the innermost xllm python frame and the innermost aten op with its
input shapes. On the 0.9B step 55% of the kernel time was elementwise / norm / copy: rope 68 ms (float,
complex multiply, cast, torch.cat, three times per layer), the q / k softmax fp32 round trips 30, the norm
weight gradients 30 (extension kernels launching 32-64 blocks), small-group rmsnorm 20, swiglu elementwise 17,
cross-entropy chunk ops 17, conv 20, layernorm 12. The 7.1B linear GEMMs run at 70-79% of peak (nvjet
256x256) and were left alone.

### Kernels added (all `xllm/modules/fused_ops/cute/`, bf16 in / out, fp32 math, Blackwell only via a
`*_supported()` check, torch path kept as the fallback)

| Kernel | File / switch | Replaces | Effect |
|---|---|---|---|
| `Rope` | `rope.py`, `XLLM_CUTE_ROPE` | `apply_rope` / `apply_ropes` (4-5 passes) | 0.9B q 547 -> 50 us; step 527 -> 469 |
| `QkPrepFwd` / `QkPrepBwd` | `qk_prep.py`, `XLLM_CUTE_QK` | rmsnorm_fwd + rope + F.softmax(fp32).to(bf16) and the recompute, rope bwd, rmsnorm_bwd, _softmax_backward_data (fp32 round trips) and two adds in the backward | one pass per direction per tensor; a warp per head row; weight-gradient partials in registers (a warp sees one head). 0.9B q: fwd 525 -> 99 us, bwd 863 -> 245; steps 0.9B 433 -> 379, 4B 381 -> 368, 7.1B 910 -> 893 |
| `AttnGateFwd` / `Bwd` | `attn_gate.py`, `XLLM_CUTE_GATE` | add + per-head rmsnorm + softplus/silu gate + multiply (10 passes) and the backward (22 passes) | 4 + 6 passes; V / 256 warps per row for V = 2048; intermediate bf16 roundings reproduced (1 ulp vs torch). 0.9B 379 -> 356, 4B 368 -> 334, 7.1B 893 -> 852 |
| `SwigluFwd` / `Bwd` | `swiglu.py`, `XLLM_CUTE_SWIGLU` | F.silu * h3 (5 passes); recompute + 2 muls + silu_backward (14 passes) | 3 + 6 passes at 6.5 TB/s, bit-identical except 1 ulp in the silu gradient. 0.9B 364 -> 352, 7.1B 813 -> 798 (in-process) |
| `CeLogits` | `cross_entropy.py`, `XLLM_CUTE_CE` | ~12 torch ops per fp32 [2048, vocab] logits chunk | one CTA per row: loss + fp32 and bf16 score matrices from the bf16 GEMM output; 1059 -> 147 us per chunk; single model-parallel rank only. 0.9B 352 -> 338 |
| `LnFwd` / `LnBwd` | `layer_norm.py`, `XLLM_CUTE_LN` | extension GroupLayerNorm fwd/bwd (1.7 TB/s) | C / 256 warps per row, 16-byte loads, partials in registers; >= 2048 tokens. [2, 16384, 1536] x 2: fwd 123 -> 54 us, bwd 197 -> 101 |
| csrc `GammaBwdLargeKernel`, `GammaBetaBwdLargeKernel` | `xllm/csrc/ops/group_{rms,layer}_norm_kernel.cu` (rebuild the extension) | the norm weight-gradient kernels: 32 x 32 blocks, channels / 32 * groups blocks (32 for a 4-head q norm), 64-byte row segments, ~0.3 TB/s | 32 x 8 blocks of 8 channels per thread (16-byte loads), row split over gridDim.z into fp32 partials + finalize; channels per group must be a multiple of 8. group_rms_norm_bwd_affine [32768, 2048] x 8: 710 -> 210 us; steps 0.9B 469 -> 433, 4B 390 -> 381, 7.1B 924 -> 910 |

SCA (`sliding_chunk_attention.py`):

- 4 x 4 sub-blocking of the causal [C x C] GEMMs of the materialized-P / dS path (`_bmm_tril_tn/nn/nt`, `_tril_sub`;
  `SCA_TRIL_N`): 7.1B no-boundary backward 4.16 -> 3.76 ms/layer. ScaDsMat never reads dP past the row's
  8-aligned diagonal, so the skipped sub-blocks stay unwritten. The kernel path's dense fp32 dPcur GEMM is
  sub-blocked the same way (`_bmm_triu_nt`, zeros below the diagonal): 1.12 -> 0.95 ms/layer.
- `ScaBwdDvT`: the fused dK+dV kernel without dK (K in TMEM as the A operand of S^T, P^T over the S^T columns,
  32-query blocks, v-slab index) for S + V > 256: S / 2 + 64 + min(V, 256) <= 512 columns. 0.9B dV 768 -> 624
  us, 7.1B 1106 -> 977. `SCA_DV_TMEM=0` restores ScaBwdDv.
- `ScaBwdDkT`: dK with the K tile in smem (TMA, A of S^T), V in TMEM (A of dP^T), double-buffered 32-query
  S^T / dP^T stages released by the compute warps, dS^T in smem: V / 2 + 128 + S <= 512 (exactly 512 for
  S = V = 256). 0.9B dK 1161 -> 922 us. `SCA_DK_TMEM=0`. First version hung: an empty-stage barrier whose
  stage is filled in the prologue must be waited with parity 0, not 1 (over-running a barrier by two phases
  deadlocks).
- `ScaBwdDqT(q_smem=True, KB=32)`: the dQ TMEM kernel with the q tile in smem and 32-key blocks (64-byte
  swizzled dS stores): V / 2 + 128 + S <= 512. 0.9B dQ 990 -> 899 us. 0.9B backward per layer 2972 -> 2489 us.
- The q / K / V -> TMEM stores need a power-of-two S / V (`Repetition`); the dispatch falls back otherwise
  (S = 192 used to fail in ScaFwdT).
- V > 256: both backward paths are compiled at the first call (`_compile_bwd_paths`). A rank compiling the GEMM
  path lazily mid-run stalled every rank of an 8-GPU step for seconds (2.6 s steps between 0.86 s steps). The
  path threshold on the attended fraction is 0.5 on one GPU and 0.35 under data parallelism, where the
  fixed-cost GEMM path avoids straggling (8 x B200 7.1B: kernel path 893, forced GEMM 870, auto 884 ms;
  `SCA_PMAT_FRAC` overrides).

### Later the same day (commits dad795e .. 202e81e)

- `ScaBwdDvT` runs 64-query blocks where K / 2 + 128 + min(V, 256) <= 512 (3 q + 2 dO stages): 0.9B dV 624
  -> 513 us, 7.1B 977 -> 813.
- `ScaBwdDkT(load_dp=True)` for V > 256: no V tile / dO blocks / dP^T MMAs, the compute warps read their key
  row of the fp32 dPcur / dPprev GEMMs per 64-query block: 7.1B dK 750 -> 485 us (kernel-path backward
  3208 -> 3055 us/layer). `ScaBwdDq` (load_dp) is the remaining old-style kernel on 7.1B (311 us).
- `ScaBwdDqT(load_dp=True)` for V > 256 (Q in TMEM, 64-key blocks, dP columns from the GEMMs): 7.1B dQ 311 ->
  159 us. The kernel-path dP GEMMs run per head on strided views of v / dout into dPcur [B, nc, H, C, C] and
  dPprev [B, nc - 1, H, C, C] (the kernels index that layout), so the two ChunkMajorCopy passes are gone:
  7.1B kernel-path backward 3394 (morning) -> 2766 us/layer; 1-GPU step 805 ms.
- The attention gate backward also emits the SCA backward's delta (rowsum of dS * y), one RowDot pass less
  per layer. Final 1-GPU steps: 0.9B 318, 4B 314, 7.1B 798 ms.
- The gate kernels were instruction bound (ncu: 51 instructions per element, math-pipe throttle): with
  beta = ln 2, softplus(x) = log2(1 + 2^x) and its derivative 1 / (1 + 2^-x) (ex2 + lg2 / rcp): forward
  116 -> 79 us (0.9B), 247 -> 167 (7.1B), same numerics. The q/k prep backward with exp2 + rcp and packed
  f32x2 rope: 245 -> 211 us. Lesson: on B200 a memory-bound elementwise kernel needs < ~10 instructions per
  element to reach bandwidth; scalar fp32 math and precise `log1p` / `1.0 / x` keep these at ~4 TB/s.
- End-to-end parity, all new paths vs all torch paths with the old extension (same batches, 3 steps): losses
  agree to 1-3e-4 and grad norms to 2e-4 on 0.9B / 4B / 7.1B (in-process steps 526 -> 316, 386 -> 274,
  907 -> 769 ms).
- 8 GPUs (per-rank batches): 7.1B 873 ms (1017 at the start of the day), 4B 358 (504), 0.9B 383;
  `fsdp_layers_per_group=4` now only buys 4 ms on the 4B. `TORCH_BLAS_PREFER_CUBLASLT=1` is slower (835 vs 813).

### Tried, not kept

- CuTe causal conv1d (width 4, bos resets, silu): the register-walk kernel became fast enough in the forward
  once the inner loop used packed f32x2 math and a mask-free path (ncu per-line view, `common/src_an.py`:
  the scalar version spent 46% of its instructions in the tap FMAs and multiplied the masks on every
  element): `cute/causal_conv.py` (commit "cute causal conv1d forward"), 0.9B q 128 -> 75 us, v (silu) 126 ->
  59, 7.1B v 207 -> 164 vs Triton; six calls per layer (the block recomputes the convs in its backward). The
  backward kernel in the module and the cp.async shared-memory staged variant (`conv/causal_conv_smem.py`) are
  correct but slower than Triton's backward even after the same treatment (4-channel threads, bf16 register
  windows, packed math, mask-free path: 186 vs 140 us on the 0.9B q shape, 473 vs 338 on 7.1B v; the dx of
  row t needs g of rows t..t+3 and g of row t needs x of rows t-3..t, so the live state per thread stays
  large): opt-in via `XLLM_CUTE_CONV_BWD=1`. Remaining: ~10 / 17 / 11 ms per step of Triton conv backward on
  0.9B / 7.1B / 4B. A different decomposition (threads along time, channels in a warp-wide shuffle) is the next
  thing to try.

### Repository test suites (script-style, run with `PYTHONPATH=<repo>:<xext dir>:<stub dir> python tests/...`)

- `tests/normalization/test_group_norm_small_groups.py`: 5040 / 5040 pass with the rebuilt extension (the first
  vectorized version failed fp32 inputs and channel counts not divisible by 8: fixed with element-size-aware
  loads and the original kernels as the fallback).
- `tests/normalization/test_rms_group_norm.py 0` / `test_layer_group_norm.py 0`: one and two mismatched elements
  out of 18 in a tiny case - identical with the original extension (pre-existing tolerance issue).
- `tests/conv/test_causal_conv1d.py 0 bf16`: 614 passes, one backward case with 1 / 7168 elements at 0.0127 vs
  0.01 tolerance - identical with `XLLM_CUTE_CONV=0` (Triton only; pre-existing).
- `tests/sca/test_sliding_chunk_attention.py` needs xattn (not installed). The CuTe suites: `tests/test_cute_*.py`
  (68 tests).

### What applies where

All block-level fusions (q/k prep, attention gate + delta, swiglu, layernorm, the fused-linear cross entropy)
live in the fused block (`GekkoBlockFunction`, `cfg.fused_block=True`, as the harness and every number here).
With `fused_block=False` only the op-level backends apply (cute SCA / AWM / TSDN as before, the conv forward,
the csrc norm kernels); the non-fused modules (`gated_delta_attention.py`, `normalized_feedforward_network.py`,
`rotary_positional_embedding.py`, `vocab_parallel_cross_entropy`) would need autograd.Function wrappers around
the same kernels.
- Fusing dS into ScaPmat (three variants, §6b) stays slower.

### Remaining (1-GPU shares)

0.9B 319 ms: SCA bwd 70 (DkT 26, DqT 25, DvT 17), SCA fwd 18, linear GEMMs ~95, conv 20, AWM 10, TSDN 5,
CE GEMMs ~8. 7.1B ~820: linear GEMMs 460 (70-79% of peak), SCA ~115 (fwd 37, dP GEMMs 30, DvT 31, Dk 24,
Dq 14), conv 34, layernorm ~12, residual adds 11, TSDN 7. 4B 323: GEMMs ~100, SCA 44, conv 18, optimizer
(outside fwd+bwd) 50, FSDP `_chunk_cat` 11 (harness).

## 7. Dev scripts (`handoff/scripts/`)

These were copied from the session scratchpad and are not polished. Most assume the repo is at `/home/johnson/xllm`.

- **`awm/bwd/`** (backward; they import the tests' `make_inputs`): `check_bwd.py <phases>` checks phases 3, 2
  (`dmem`), 1 against the FP64 spec; `prof_bwd.py <shape>` per-kernel GPU times (torch.profiler) of the saved-
  workspace path; `bench_bwd.py` event timing; `diag_dk2.py` repeated-run determinism; `ncu_bwd.py` +
  `ncu_lines.py` per-source-line stall samples (`ncu --page source --print-source cuda,sass --csv` with
  `CUTE_DSL_LINEINFO=1`); `recprobe/` globaltimer timeline probes (they need a copy of the module with the probe
  lines; the generators lived in the session, see the memory notes).
- **`awm/`**
  - `check_awm_fwd.py` provides `make` and `as_blhs_views` and is imported by the others.
  - `bench_awm_ev.py` measures back-to-back event time; `bench_awm_fwd.py` and `breakdown_awm.py` give the per-kernel split.
  - `run_gekko.py <preset> <B> <L>` runs a full training step. Set `AWM_BACKEND=cute|torch`.
  - `ncu_awm.py` is the driver for ncu.
  - `awm_proto.py` and `awm_bwd_proto.py` are the phase-decomposition prototypes.
  - `recprobe/` holds the globaltimer timeline probe.
- **`sca/`**: `test_fwd.py` / `test_bwd.py <B,H,K,S,V,L,C,seg_p[,nocheck]> ...` check the SCA kernels against
  the dense reference (autograd for the backward) and time them; `prof_bwd.py <shape>` per-kernel times of the
  backward; `layout_check.py` prints the K-major and MN-major SW128 smem layouts (they coincide).
- **`profile/`**: `prof_step.py <preset> <B> <L> [trace.json]` per-category kernel times of a training step;
  `prof_stack2.py <preset> <B> <L> [kernel regex]` writes a chrome trace with python stacks and `attrib.py
  <trace> <regex> <depth>` attributes kernel time to xllm call sites (launches via cuda_runtime and cuda_driver);
  `fsdp_step.py` (torchrun) FSDP2 steps with clip + optimizer and a kernel-class breakdown on rank 0;
  `parity_sca.py <preset> <B> <L>` swift vs cute SCA losses / grad norms / step time / peak memory. Env:
  `TN_BACKEND`, `AWM_BACKEND`, `CFG_<model field>=value`. For §6c: `run1.sh <gpus> <nproc> <port> <trace>
  <preset> <B> <L> [steps]` wraps `fsdp_trace.py` (env `FSDP_FP32_RS`, `FSDP_RESHARD`, `SAME_SEED`,
  `SEED_OFFSET`, `PROF_STEP`, `WITH_STACK`, `TRACE_OUT`); `gaps.py <trace> [top]` compute-stream gaps and NCCL
  overlap inside the FWDBWD window; `gapattr.py <trace> [depth] [regex] [--ops]` GPU-idle time by python frame
  or cpu op; `cpuself.py <trace> [top] [--fwd]` CPU self time per python function; `kcmp.py <a> <b>` kernel
  time diff of two traces.
- **`fusions/`** (§6d): `test_rope.py`, `test_qk.py` / `qk_var.py`, `test_gate.py`, `test_swiglu.py`, `test_ce.py`,
  `test_ln.py` compare each fused kernel with the torch / extension path it replaces (max error, fp32
  reference, timing at the real shapes); `subblk.py` times the sub-blocked causal GEMMs. `profile/opattr.py
  <trace> <regex>` kernel time by python frame + aten op + input shapes (trace from `run1.sh` with
  `WITH_STACK=1 RECORD_SHAPES=1`); `profile/parity_env.py <preset> <B> <L> <ENV> <v1,v2> [xext]` loss /
  grad-norm / per-param parity between values of an env var; `profile/gamma_check.py <xext dir>` checks and
  times the extension's norm backward (old vs rebuilt .so). `run1.sh` / `fsdp_trace.py` take `XEXT` (extension
  dir) and `RECORD_SHAPES`; `prof_step.py` honours `TN_BACKEND` and `CFG_<field>`.
- **`conv/`**: the two conv1d drafts and their tests (`test_conv.py`, `test_conv_s.py` vs the reference and
  Triton kernels), `conv_once.py` (ncu driver), `conv_bisect.py` (CPU-only compile bisect of DSL errors).
- **`tsdn/`**: check, bench, sweep and profile scripts for TSDN. `cubext/` builds the CUB baseline.
- **`common/ncu_src.sh <tag> [shape] [cfg]`**: source-level ncu stall breakdown. Set `NCU`, `KNAME` (kernel regex) and `SCRIPT` (a driver such as `../awm/ncu_awm.py`), plus `SRC` and `NLINES`. It uses `CUTE_DSL_LINEINFO=1` and `--import-source yes`. `src_an.py` aggregates per source line.

Not copied, because they are large or can be regenerated:

- ncu reports
- pre-experiment kernel snapshots
- the downloaded wheels and .deb
- the built extension
- the reference sources: FA4 `flash_attn/cute`, quack 0.6.5, and the CUTLASS CuTeDSL Blackwell tutorial GEMMs
- the Gecko paper, arXiv 2601.06463

## 8. Lessons (details in `handoff/memory/`)

### CuTe DSL 4.8.1 pitfalls (additions from §6d)

- `cute.make_rmem_tensor(layout, dtype)` is the register array (no `make_fragment`); index elements for
  pair-wise math (rope), `.load()` / `.store()` for vector ops.
- Python `and` / `or` on DSL values inside kernels breaks IR dominance: use `&` on comparisons. A `.load()` SSA
  vector consumed in a loop under a `const_expr` branch failed the same way: stage it in an rmem tensor.
- `exp` / `rsqrt` with `fastmath=False` are slow software sequences; `fastmath=True` (MUFU) is what the torch
  kernels effectively use.
- Vector loads need `.align(bytes)` that is a power of two (8 elements bf16 = 16 bytes).
- Every process compiles every kernel it uses (seconds each): in multi-rank runs pre-compile data-dependent
  paths, or a rank's first use stalls the whole step.

### CuTe DSL 4.8.1 pitfalls

- `PipelineTmaAsync` consumer release only arrives from the signaling thread. For multi-warp consumers, pass `tidx=lane` with `enable_multicast_signaling=True`.
- Every TMEM user must wait on the alloc barrier before its main loop, or it deadlocks.
- Build the t2r destination from `partition_D(identity)`.
- TMA `cute.copy` must be called from a full warp. Never wrap it in `if tidx==0`, which hangs.
- A Python `for` over `range` in a kernel becomes dynamic. Use `range_constexpr` or a list comprehension.
- Do not reuse a loop variable name with different types.
- `range_constexpr` is not allowed in non-jit helpers.

### Tried with no gain on the AWM forward

- PDL across the 6 kernels
- 8 transform warps
- multicasting B across a 2-CTA cluster
- polynomial exp2 in lse
- cp.async block staging in lse
- VT=128 in the tcgen05 recurrence
- B prefetch in the recurrence
- a deeper q ring in the out GEMM

### Host overhead

The forward is 6 TVM-FFI launches, about 118 µs on the host. That is hidden in training, but it bounds 4B back-to-back.
