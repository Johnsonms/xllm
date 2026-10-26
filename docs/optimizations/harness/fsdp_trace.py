"""FSDP2 Gekko training steps (fwd+bwd+clip+optim) on all torchrun ranks; rank 0 reports timings and a kernel breakdown.

usage: run1.sh <gpus> <nproc> <port> <trace|/dev/null> <preset> <batch_per_gpu> <seq_len> [steps=5]  (torchrun fsdp_trace.py ...)
"""
import os, sys, time, re, json
from collections import defaultdict
sys.path.insert(0, os.environ.get("XEXT", os.path.expanduser("~/scratch/xext")))  # xllm_extension build dir
sys.path.insert(0, os.environ.get("XLLM_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
import torch, torch.distributed as dist, types
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
preset, bsz, seq_len = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 5
local_rank = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(local_rank)
if os.environ.get("SM_CARVEOUT"): torch._C._set_sm_carveout_experimental(int(os.environ["SM_CARVEOUT"]))
if os.environ.get("PIN_CPU"):  # pin the rank to PIN_CPU cores of its GPU's NUMA node (PIN_GPUS_PER_NODE GPUs and PIN_CPUS_PER_NODE cores per node)
    _per, _gpn, _cpn = int(os.environ["PIN_CPU"]), int(os.environ.get("PIN_GPUS_PER_NODE", "4")), int(os.environ.get("PIN_CPUS_PER_NODE", "78"))
    _node, _i = local_rank // _gpn, local_rank % _gpn
    os.sched_setaffinity(0, range(_node * _cpn + _i * _per, _node * _cpn + (_i + 1) * _per))
from xllm.distributed import init_torch_distributed, initialize_model_parallel
init_torch_distributed(1800); initialize_model_parallel(1, 1, 1800)
rank, world = dist.get_rank(), dist.get_world_size()
import xllm.modules.fused_ops
from xllm.models import build_model
from xllm.optim import build_optimizer
from xllm.utils import clip_grad_norm_
from xllm.config import OptimConf
from xllm.configuration import ConfStore
class Tok: bos_id, eos_id, pad_id, vocab_size, n_words = 1, 2, 0, int(os.environ.get("VOCAB", "32000")), int(os.environ.get("VOCAB", "32000"))
cfg = ConfStore[preset]; cfg.vocab_size = Tok.vocab_size; cfg.fused_block = True
cfg.awm_backend = os.environ.get("AWM_BACKEND", "cute"); cfg.timenorm_backend = os.environ.get("TN_BACKEND", "cub")
for k, v in os.environ.items():
    if k.startswith("CFG_"):
        cur = getattr(cfg, k[4:]); setattr(cfg, k[4:], (v == "1") if isinstance(cur, bool) else type(cur)(v))
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
# FSDP_SHARD_SIZE = HSDP shard group size (1 = pure replication); unset = fully sharded over the world
model = build_model(cfg, dtype="bf16", fully_sharded_size=int(os.environ["FSDP_SHARD_SIZE"]) if os.environ.get("FSDP_SHARD_SIZE") else None, fp32_reduce_scatter=bool(int(os.environ.get("FSDP_FP32_RS", "1"))), reshard_after_forward=bool(int(os.environ.get("FSDP_RESHARD", "1"))), forward_prefetch=bool(int(os.environ.get("FSDP_FWD_PREFETCH", "0"))), tokenizer=Tok())
if os.environ.get("FSDP_SYMM"):  # NCCL symmetric-memory all-gather staging (copy-engine all-gather with NCCL_CTA_POLICY=2)
    from torch.distributed.fsdp import FSDPModule
    for _m in model.modules():
        if isinstance(_m, FSDPModule):
            if os.environ.get("FSDP_SUMRS"): _m.set_force_sum_reduction_for_comms(True)  # symmetric reduce-scatter (SUM only)
            _m.set_symm_mem_for_comm()
if os.environ.get("FSDP_PGALLOC"):  # NCCL-allocated staging buffers for all-gather and reduce-scatter
    from torch.distributed.fsdp import FSDPModule
    for _m in model.modules():
        if isinstance(_m, FSDPModule): _m.set_allocate_memory_from_process_group_for_comm(True)
if int(os.environ.get("FSDP_PREFETCH_DEPTH", "0")) > 0:  # explicit FSDP2 forward prefetch of the next N units
    from torch.distributed.fsdp import FSDPModule
    _units = [m for m in model.modules() if isinstance(m, FSDPModule) and m is not model]
    _d = int(os.environ["FSDP_PREFETCH_DEPTH"])
    for _i, _m in enumerate(_units):
        if _i + 1 < len(_units): _m.set_modules_to_forward_prefetch(_units[_i + 1:_i + 1 + _d])
    if rank == 0: print(f"forward prefetch depth {_d} over {len(_units)} FSDP units", flush=True)
optimizer, scheduler = build_optimizer(model, OptimConf(fused=os.environ.get("ADAM_FOREACH", "0") != "1"), 1000, "bf16")
g = torch.Generator(device="cuda").manual_seed(1000 + int(os.environ.get("SEED_OFFSET", "0")) + (0 if os.environ.get("SAME_SEED") else rank))
def batch():
    tokens = torch.randint(3, Tok.vocab_size, (bsz, seq_len + 1), device="cuda", generator=g)
    tokens[torch.rand(bsz, seq_len + 1, device="cuda", generator=g) < float(os.environ.get("EOS_RATE", 1 / 3000))] = Tok.eos_id
    return tokens[:, :-1], tokens[:, 1:]
def step(prof=None):
    x, y = batch()
    if os.environ.get("BALANCE") == "1":
        from xllm.data.balance import balance_rows_across_ranks
        x, y, _ = balance_rows_across_ranks(x, y, None, Tok.eos_id, cfg.chunk_size)
    torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
    with torch.profiler.record_function("FWDBWD"):
        tok_loss, aux_loss, _ = model(tokens=x, targets=y, token_mask=None, multi_segments=True, deterministic=False)
        loss = tok_loss.sum() / tok_loss.numel()
        loss.backward()
        torch.cuda.synchronize()
    t1 = time.time()
    if os.environ.get("NANCHECK") == "1":
        bad = [n for n, p_ in model.named_parameters() if p_.grad is not None and not torch.isfinite(p_.grad.to_local() if hasattr(p_.grad, "to_local") else p_.grad).all()]
        if bad:
            print(f"NAN rank {rank} step {len(res)}: {len(bad)} params, first {bad[:3]} last {bad[-3:]}", flush=True)
        flag = torch.tensor([float(bool(bad))], device="cuda"); dist.all_reduce(flag)
        if flag.item() > 0:
            optimizer.zero_grad()
            return float("nan"), float("nan"), (t1 - t0) * 1000, 0.0, 0.0
    gn = clip_grad_norm_(fsdp_module=model, max_norm=1.0)
    torch.cuda.synchronize(); t2 = time.time()
    optimizer.step(); scheduler.step(); optimizer.zero_grad()
    torch.cuda.synchronize(); t3 = time.time()
    return loss.item(), float(gn), (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000
res = []
for s in range(steps):
    if s == int(os.environ.get("PROF_STEP", steps - 1)) and rank == 0:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU], with_stack=bool(int(os.environ.get("WITH_STACK", "0"))), record_shapes=bool(int(os.environ.get("RECORD_SHAPES", "0")))) as prof:
            r = step(); torch.cuda.synchronize()
        prof.export_chrome_trace(os.environ.get("TRACE_OUT", f"fsdp_{preset}_w{world}.json"))
    else:
        r = step()
    res.append(r)
    if rank == 0: print(f"step {s}: loss {r[0]:.4f} gnorm {r[1]:.4f}  fwd+bwd {r[2]:.0f} ms  clip {r[3]:.0f} ms  optim {r[4]:.0f} ms", flush=True)
if rank == 0:
    mid = res[1:-1] if steps > 2 else res
    print(f"{preset} world={world} B={bsz} L={seq_len} fp32rs={os.environ.get('FSDP_FP32_RS','1')} reshard={os.environ.get('FSDP_RESHARD','1')} grp={cfg.fsdp_layers_per_group} sameseed={bool(os.environ.get('SAME_SEED'))}: mean fwd+bwd {sum(r[2] for r in mid)/len(mid):.0f} ms, clip {sum(r[3] for r in mid)/len(mid):.0f} ms, optim {sum(r[4] for r in mid)/len(mid):.0f} ms; peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    mean = sum(r[2] + r[3] + r[4] for r in mid) / len(mid)
    med = sorted(r[2] for r in mid)[len(mid) // 2]
    print(f"  tokens/s/GPU {bsz * seq_len / (mean / 1000):.0f} (fwd+bwd+clip+optim mean {mean:.0f} ms); median fwd+bwd {med:.0f} ms", flush=True)
    if int(os.environ.get("PROF_STEP", steps - 1)) < 0:  # no profiled step
        dist.barrier(); dist.destroy_process_group(); sys.exit(0)
    ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA and e.name not in ("FWD", "BWD")]
    first = min(e.time_range.start for e in ev); last = max(e.time_range.end for e in ev)
    cats = defaultdict(float)
    for e in ev:
        n = e.name
        c = "nccl" if "nccl" in n.lower() or "ncclDevKernel" in n else ("gemm" if re.search("gemm|nvjet|cutlass", n) else ("copy" if re.search("copy|Copy|Cat", n) else ("sca" if "Attention" in n else ("optim" if re.search("adam|Adam|multi_tensor|foreach|lerp", n) else "other"))))
        cats[c] += e.device_time / 1000
    # GPU busy time (union of kernel intervals, nccl kernels excluded and included)
    def busy(evs):
        iv = sorted((e.time_range.start, e.time_range.end) for e in evs); tot = 0; cs, ce = None, None
        for a, b in iv:
            if cs is None or a > ce:
                if cs is not None: tot += ce - cs
                cs, ce = a, b
            else: ce = max(ce, b)
        if cs is not None: tot += ce - cs
        return tot / 1000
    print(f"  profiled last step: span {(last-first)/1000:.0f} ms; busy(all kernels) {busy(ev):.0f} ms; busy(non-nccl) {busy([e for e in ev if 'nccl' not in e.name.lower()]):.0f} ms")
    print("  kernel time by class (ms): " + ", ".join(f"{k}={v:.0f}" for k, v in sorted(cats.items(), key=lambda x: -x[1])))
dist.barrier(); dist.destroy_process_group()
