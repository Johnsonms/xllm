"""FSDP2 Gekko training steps (fwd+bwd+clip+optim) on all torchrun ranks; rank 0 reports timings and a kernel breakdown.

usage: torchrun --nproc_per_node N fsdp_step.py <preset> <batch_per_gpu> <seq_len> [steps=5]
"""
import os, sys, time, re, json
from collections import defaultdict
S = "/tmp/claude-243000396/-data-home-johnson-xllm/d3839bfb-7816-4668-9945-7c337f0c3f9e/scratchpad/handoff/scripts/awm"
sys.path.insert(0, os.path.join(S, "xext")); sys.path.insert(0, "/home/johnson/xllm")
import torch, torch.distributed as dist, types
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
preset, bsz, seq_len = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 5
local_rank = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(local_rank)
from xllm.distributed import init_torch_distributed, initialize_model_parallel
init_torch_distributed(1800); initialize_model_parallel(1, 1, 1800)
rank, world = dist.get_rank(), dist.get_world_size()
import xllm.modules.fused_ops
from xllm.models import build_model
from xllm.optim import build_optimizer
from xllm.utils import clip_grad_norm_
from xllm.config import OptimConf
from xllm.configuration import ConfStore
class Tok: bos_id, eos_id, pad_id, vocab_size, n_words = 1, 2, 0, 32000, 32000
cfg = ConfStore[preset]; cfg.vocab_size = Tok.vocab_size; cfg.fused_block = True
cfg.awm_backend = os.environ.get("AWM_BACKEND", "cute"); cfg.timenorm_backend = os.environ.get("TN_BACKEND", "cub")
for k, v in os.environ.items():
    if k.startswith("CFG_"):
        cur = getattr(cfg, k[4:]); setattr(cfg, k[4:], (v == "1") if isinstance(cur, bool) else type(cur)(v))
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
model = build_model(cfg, dtype="bf16", fully_sharded_size=None, fp32_reduce_scatter=True, reshard_after_forward=True, forward_prefetch=False, tokenizer=Tok())
optimizer, scheduler = build_optimizer(model, OptimConf(), 1000, "bf16")
g = torch.Generator(device="cuda").manual_seed(1000 + rank)
def batch():
    tokens = torch.randint(3, Tok.vocab_size, (bsz, seq_len + 1), device="cuda", generator=g)
    tokens[torch.rand(bsz, seq_len + 1, device="cuda", generator=g) < 1 / 3000] = Tok.eos_id
    return tokens[:, :-1], tokens[:, 1:]
def step(prof=None):
    x, y = batch()
    torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
    tok_loss, aux_loss, _ = model(tokens=x, targets=y, token_mask=None, multi_segments=True, deterministic=False)
    loss = tok_loss.sum() / tok_loss.numel()
    loss.backward()
    torch.cuda.synchronize(); t1 = time.time()
    gn = clip_grad_norm_(fsdp_module=model, max_norm=1.0)
    torch.cuda.synchronize(); t2 = time.time()
    optimizer.step(); scheduler.step(); optimizer.zero_grad()
    torch.cuda.synchronize(); t3 = time.time()
    return loss.item(), float(gn), (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000
res = []
for s in range(steps):
    if s == steps - 1 and rank == 0:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as prof:
            r = step(); torch.cuda.synchronize()
    else:
        r = step()
    res.append(r)
    if rank == 0: print(f"step {s}: loss {r[0]:.4f} gnorm {r[1]:.4f}  fwd+bwd {r[2]:.0f} ms  clip {r[3]:.0f} ms  optim {r[4]:.0f} ms", flush=True)
if rank == 0:
    mid = res[1:-1] if steps > 2 else res
    print(f"{preset} world={world} B={bsz} L={seq_len} awm={cfg.awm_backend} tn={cfg.timenorm_backend}: mean fwd+bwd {sum(r[2] for r in mid)/len(mid):.0f} ms, clip {sum(r[3] for r in mid)/len(mid):.0f} ms, optim {sum(r[4] for r in mid)/len(mid):.0f} ms; peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
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
