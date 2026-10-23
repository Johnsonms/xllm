# CuTe DSL kernels for xllm: handoff

This handoff covers two tasks:

- `TimestepDecayNorm` (TSDN): forward and backward are done.
- `AdaptiveWorkingMemory` (AWM): forward and backward are done (section 6 has what is left).

Both are written in the CuTe Python DSL (nvidia-cutlass-dsl 4.8.1) and target B200 (sm_100a).

The original request came from Max: replace the CUB/torch implementations with FA-level-performance CuTe kernels. Use ncu to find bottlenecks. Push finished commits to `origin` (github.com/Johnsonms/xllm).

## 1. Branches (all pushed to origin)

| Branch | Content |
|---|---|
| `cute-timestep-decay-norm` | TSDN fwd+bwd kernels, tests, and the `timenorm_backend='cute'` wiring (10 commits on `main`). |
| `cute-adaptive-working-memory` | The TSDN branch plus the AWM forward and backward kernels, tests and the `awm_backend='cute'` wiring. **This is the working branch.** |
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
  placed with pointer offsets (wrong results: the second use of a half read the first use's data - avoid
  `tmem_ptr + offset`, use a stacked fragment); issuing the last raw key blocks before the GEMM loads (slower).
  Next: a 2-CTA (cta_group::2) dP^T GEMM halves the B bytes per CTA; 7.1B (K = 2304) would gain most.
- `AwmDmoGemm` is at ~56% of HBM SOL; `AwmOutGemm` (dq) is the forward kernel.
- `AwmRecurrenceBwd`: ~10 us per update; the dA GEMM was moved out of the chain (batched matmul).
- Feature gap: context-parallel ranks > 0 (input memory and previous chunk) use the torch backward. The kernels
  have the hooks (`first`, `has_prev`, `has_dmem`, `_seg_anchor`) but the chunk kernel reads the update chunk
  from the current sequence only and the previous-chunk gradients are not produced.

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
