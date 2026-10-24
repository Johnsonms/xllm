"""Per-kernel GPU-time breakdown of one Gekko training step (forward / backward), grouped by category.

usage: AWM_BACKEND=cute python prof_step.py <preset> <batch> <seq_len> [trace.json]
"""
import os, sys, time, re, json
from collections import defaultdict
S = "/tmp/claude-243000396/-data-home-johnson-xllm/d3839bfb-7816-4668-9945-7c337f0c3f9e/scratchpad/handoff/scripts/awm"
sys.path.insert(0, os.path.join(S, "xext")); sys.path.insert(0, "/home/johnson/xllm")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29544")
import torch, torch.distributed as dist, types
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
preset, bsz, seq_len = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
trace = sys.argv[4] if len(sys.argv) > 4 else None
dist.init_process_group("nccl", world_size=1, rank=0); torch.cuda.set_device(0)
from xllm.distributed import initialize_model_parallel
initialize_model_parallel(1, 1, 1800)
import xllm.modules.fused_ops
from xllm.models.gekko import Gekko
from xllm.configuration import ConfStore
class Tok: bos_id, eos_id, pad_id, vocab_size, n_words = 1, 2, 0, 32000, 32000
cfg = ConfStore[preset]; cfg.vocab_size = Tok.vocab_size; cfg.fused_block = True
cfg.awm_backend = os.environ.get("AWM_BACKEND", "cute"); cfg.timenorm_backend = os.environ.get("TN_BACKEND", "cub")
for k, v in os.environ.items():
    if k.startswith("CFG_"):
        cur = getattr(cfg, k[4:]); setattr(cfg, k[4:], (v == "1") if isinstance(cur, bool) else type(cur)(v))
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)
model = Gekko(cfg, Tok()).cuda().to(torch.bfloat16); model.train()
print(f"{preset}: H={cfg.num_heads} K={cfg.num_kv_heads} S={cfg.head_dim} V={cfg.v_head_dim} C={cfg.chunk_size} dim={cfg.model_dim} "
      f"layers={cfg.num_layers} B={bsz} L={seq_len} awm={cfg.awm_backend}")
for k in sorted(vars(cfg)):
    if re.search("recompute|checkpoint|dropout|ffn|hidden|swiglu|attn|chunk|conv|norm|tsdn|gate", k):
        print(f"   cfg.{k} = {getattr(cfg, k)}")
g = torch.Generator(device="cuda").manual_seed(0)
def step():
    tokens = torch.randint(3, Tok.vocab_size, (bsz, seq_len + 1), device="cuda", generator=g)
    tokens[torch.rand(bsz, seq_len + 1, device="cuda", generator=g) < 1 / 3000] = Tok.eos_id
    x, y = tokens[:, :-1], tokens[:, 1:]
    with torch.profiler.record_function("FWD"):
        loss = model(x, True, targets=y); loss = loss[0] if isinstance(loss, tuple) else loss
    with torch.profiler.record_function("BWD"):
        loss.float().mean().backward()
    model.zero_grad(set_to_none=True)
for _ in range(2): step()
torch.cuda.synchronize(); t0 = time.time(); step(); torch.cuda.synchronize(); wall = (time.time() - t0) * 1000
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as prof:
    step(); torch.cuda.synchronize()
if trace: prof.export_chrome_trace(trace)
ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
cpu = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CPU and e.name in ("FWD", "BWD")]
# kernels inside FWD / BWD by launch time: correlate via the profiler's chrome trace instead (simpler: time ranges)
kern_total = sum(e.device_time for e in ev) / 1000
first = min(e.time_range.start for e in ev); last = max(e.time_range.end for e in ev)
print(f"\nwall {wall:.0f} ms; kernel sum {kern_total:.0f} ms; kernel span {(last-first)/1000:.0f} ms")
CATS = [("awm cute", r"awm|Awm|adaptive_working"), ("gemm", r"gemm|nvjet|cutlass|Gemm|GEMM|splitK|xmma|wgrad|sgemm|hgemm|bgemm|matmul"),
        ("sliding chunk attn", r"swift|flash|fmha|attn|sdpa|Attention|attention"), ("timestep norm", r"timenorm|timestep|TimeStep|tsn|cumsum|scan"),
        ("conv", r"conv"), ("rmsnorm/layernorm", r"rms|RMS|norm|Norm"), ("softmax", r"softmax|Softmax"),
        ("cross entropy", r"cross_entropy|CrossEntropy|log_softmax|nll"), ("dropout", r"dropout|philox|rng|bernoulli"),
        ("swiglu/silu", r"swiglu|silu|Silu|SiLU|glu"), ("reduce", r"reduce|Reduce|sum|Sum|norm_kernel"), ("copy/cat/fill", r"copy|Copy|cat|Cat|fill|Fill|Index|index|gather|scatter|transpose"),
        ("elementwise", r"elementwise|vectorized|unrolled|binary|Binary|unary|Unary|Functor|Op")]
def cat(n):
    for c, pat in CATS:
        if re.search(pat, n): return c
    return "other"
by_cat = defaultdict(float); by_name = defaultdict(float); cnt = defaultdict(int)
for e in ev:
    by_cat[cat(e.name)] += e.device_time / 1000; by_name[e.name] += e.device_time / 1000; cnt[e.name] += 1
print("\nby category (ms, % of kernel sum):")
for c, t in sorted(by_cat.items(), key=lambda x: -x[1]):
    print(f"  {t:8.1f}  {100*t/kern_total:5.1f}%  {c}")
print("\ntop kernels:")
for n, t in sorted(by_name.items(), key=lambda x: -x[1])[:45]:
    print(f"  {t:8.1f}  {100*t/kern_total:5.1f}%  x{cnt[n]:4d}  [{cat(n)}] {n[:120]}")
# forward / backward split from the FWD/BWD CPU ranges: kernels are attributed by their launch time via correlation in the trace
if trace:
    with open(trace) as f: tr = json.load(f)["traceEvents"]
    ranges = {e["name"]: (e["ts"], e["ts"] + e["dur"]) for e in tr if e.get("name") in ("FWD", "BWD") and e.get("cat") in ("user_annotation", "cpu_op")}
    launch = {}
    for e in tr:
        if e.get("cat") == "cuda_runtime" and "args" in e and "correlation" in e["args"]:
            launch[e["args"]["correlation"]] = e["ts"]
    split = defaultdict(lambda: defaultdict(float))
    for e in tr:
        if e.get("cat") == "kernel":
            ts = launch.get(e["args"].get("correlation"), e["ts"])
            ph = next((k for k, (a, b) in ranges.items() if a <= ts <= b), "?")
            split[ph][cat(e["name"])] += e["dur"] / 1000
    for ph in ("FWD", "BWD", "?"):
        tot = sum(split[ph].values())
        print(f"\n{ph}: {tot:.0f} ms")
        for c, t in sorted(split[ph].items(), key=lambda x: -x[1]):
            print(f"  {t:8.1f}  {c}")
print(f"\npeak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
dist.destroy_process_group()
