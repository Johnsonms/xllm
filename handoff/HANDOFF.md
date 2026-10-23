# CuTe DSL kernels for xllm: handoff

This handoff covers two tasks:

- `TimestepDecayNorm` (TSDN): forward and backward are done.
- `AdaptiveWorkingMemory` (AWM): the forward is done; the backward is next.

Both are written in the CuTe Python DSL (nvidia-cutlass-dsl 4.8.1) and target B200 (sm_100a).

The original request came from Max: replace the CUB/torch implementations with FA-level-performance CuTe kernels. Use ncu to find bottlenecks. Push finished commits to `origin` (github.com/Johnsonms/xllm).

## 1. Branches (all pushed to origin)

| Branch | Content |
|---|---|
| `cute-timestep-decay-norm` | TSDN fwd+bwd kernels, tests, and the `timenorm_backend='cute'` wiring (10 commits on `main`). |
| `cute-adaptive-working-memory` | The TSDN branch plus the AWM forward kernels, tests, `awm_backend='cute'` wiring, and the backward spec. Head `1165bc7`. **This is the working branch.** |
| `cute-kernels-handoff` | `cute-adaptive-working-memory` plus this `handoff/` directory (dev scripts, notes, venv freeze). Do not merge it; it is for reference only. |

Commit convention (no git identity is configured on the machines):

```
git -c user.name="Johnson" -c user.email="johnson@together.ai" commit ...
# message ends with:
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```

## 2. Environment setup on a new node

The steps below assume 8× B200, driver with CUDA 13, and system python 3.10. The repo lives at `~/xllm` and the venv at `~/xllm-venv`. Many scratch scripts hardcode `/home/johnson/xllm`; clone there, or symlink it.

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
~/xllm-venv/bin/python -m pytest tests/test_cute_adaptive_working_memory.py -q          # 12 pass (9 fwd + 3 phased-bwd spec)
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

## 6. Next step: AWM backward kernels

Training currently spends about 36 ms per step on the torch recompute (`_accum_fwd`) and about 59 ms on the torch backward (0.9B). Follow `awm_bwd_phased` in `adaptive_working_memory_reference.py`, and the prototype `handoff/scripts/awm/awm_bwd_proto.py`:

- **3′. `dq_c = dOut_c · M_cᵀ` and `dMo_c = Σ_h q_cᵀ · dOut_c`.** These are parallel over chunks and have the same structure as `AwmOutGemm`, so reuse it. M_c must be saved by the forward (or recomputed by the recurrence), so the forward needs a "save M sequence" option.
- **2′. Reverse recurrence over the updates.** Set D = B − A·M − M and dr = Σ_v G∘D. Then:
  - dB = r·G
  - dA = −(r·G)·Mᵀ
  - G_prev = (1−r)·G − Aᵀ·(r·G) + dMo
  - the lse and lnt chain through logaddexp, plus memory-reset masking.

  This is sequential like the forward recurrence, so the two `AwmRecurrence*` kernels are the template.
- **1′. Per-chunk kernel, parallel.** Compute:
  - dqk = Pᵀ·dA
  - dv = Pᵀ·dB
  - dP = dA·qkᵀ + dB·vᵀ
  - dk = P∘(dP − rowsum(P∘dP)) + P·dlse

  P is recomputed from k and lse, as in the forward. Use the `AwmChunkGemm` structure for this.
- Then wire it as `awm_backend='cute'` in the backward of `AdaptiveWorkingMemoryFunc` and the fused block, and validate the end-to-end loss and gradients with `run_gekko.py`.

## 7. Dev scripts (`handoff/scripts/`)

These were copied from the session scratchpad and are not polished. Most assume the repo is at `/home/johnson/xllm`.

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
