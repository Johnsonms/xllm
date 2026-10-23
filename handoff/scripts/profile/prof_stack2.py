"""Attribute GPU kernel time of one Gekko step to the deepest enclosing python frame in xllm (chrome trace sweep)."""
import os, sys, re, json, bisect
from collections import defaultdict
S = "/tmp/claude-243000396/-data-home-johnson-xllm/d3839bfb-7816-4668-9945-7c337f0c3f9e/scratchpad/handoff/scripts/awm"
sys.path.insert(0, os.path.join(S, "xext")); sys.path.insert(0, "/home/johnson/xllm")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
import torch, torch.distributed as dist, types
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
preset, bsz, seq_len = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
pat = sys.argv[4] if len(sys.argv) > 4 else "."
trace = f"stack_{preset}.json"
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
torch.manual_seed(0)
model = Gekko(cfg, Tok()).cuda().to(torch.bfloat16); model.train()
g = torch.Generator(device="cuda").manual_seed(0)
def step():
    tokens = torch.randint(3, Tok.vocab_size, (bsz, seq_len + 1), device="cuda", generator=g)
    tokens[torch.rand(bsz, seq_len + 1, device="cuda", generator=g) < 1 / 3000] = Tok.eos_id
    x, y = tokens[:, :-1], tokens[:, 1:]
    loss = model(x, True, targets=y); loss = loss[0] if isinstance(loss, tuple) else loss
    loss.float().mean().backward(); model.zero_grad(set_to_none=True)
for _ in range(2): step()
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU], with_stack=True) as prof:
    step(); torch.cuda.synchronize()
prof.export_chrome_trace(trace)
dist.destroy_process_group()
tr = json.load(open(trace))["traceEvents"]
def short(n): return re.sub(r"^.*?/xllm/", "", n)
frames = sorted([(e["ts"], e["ts"] + e["dur"], short(e["name"])) for e in tr if e.get("cat") == "python_function" and "/xllm/" in e.get("name", "") and "site-packages" not in e["name"] and "scratchpad" not in e["name"]])
launch = {e["args"]["correlation"]: e["ts"] for e in tr if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})}
kern = sorted([(launch.get(e["args"].get("correlation"), -1), e["dur"], e["name"]) for e in tr if e.get("cat") == "kernel"])
tot = defaultdict(float); kn = defaultdict(lambda: defaultdict(float)); alltot = 0.0; fi = 0; stack = []
for ts, dur, name in kern:
    alltot += dur
    while fi < len(frames) and frames[fi][0] <= ts:
        while stack and stack[-1][1] <= frames[fi][0]: stack.pop()
        stack.append(frames[fi]); fi += 1
    while stack and stack[-1][1] < ts: stack.pop()
    if not re.search(pat, name): continue
    key = " <- ".join(f[2] for f in reversed(stack[-3:])) if stack else "?"
    tot[key] += dur / 1000; kn[key][name[:80]] += dur / 1000
print(f"{preset}: kernels matching '{pat}': {sum(tot.values()):.0f} ms of {alltot/1000:.0f} ms")
for s, t in sorted(tot.items(), key=lambda x: -x[1])[:45]:
    print(f"{t:7.1f} ms  {s}")
    for n, tt in sorted(kn[s].items(), key=lambda x: -x[1])[:3]:
        print(f"           {tt:6.1f}  {n}")
