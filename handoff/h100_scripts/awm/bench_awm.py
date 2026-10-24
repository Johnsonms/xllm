"""cute AWM per call on the current GPU at the Gekko training shapes (L = 16384, chunk 4096, segment_idx): GPU time
of the forward (with workspaces) and the backward from the workspaces, with the top kernels.
usage: [AWM_SM90_CUBLAS=1] python bench_awm.py [top_kernels=6]"""
import sys, types, importlib
import torch
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
from torch.profiler import profile, ProfilerActivity
cfwd = importlib.import_module("xllm.modules.fused_ops.cute.adaptive_working_memory")
cbwd = importlib.import_module("xllm.modules.fused_ops.cute.adaptive_working_memory_bwd")

SHAPES = [("0.9B", 2, 8, 4, 256, 256), ("4B", 1, 32, 8, 128, 128), ("7.1B", 1, 4, 4, 256, 2048)]
L, C = 16384, 4096
TOP = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def prof(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    ev = [e for e in p.key_averages() if e.device_time_total > 0]
    tot = sum(e.device_time_total for e in ev) / iters
    top = sorted(ev, key=lambda e: -e.device_time_total)[:TOP]
    return tot, [(e.key[:60], e.device_time_total / iters) for e in top]


for name, B, H, K, S, V in SHAPES:
    g = torch.Generator(device="cuda").manual_seed(0)
    rn = lambda *s: torch.randn(*s, device="cuda", generator=g)
    q = torch.softmax(rn(B, L, H, S) * 2, -1).bfloat16().transpose(1, 2)
    k = (rn(B, L, K, S) * 2).bfloat16()
    qk = torch.softmax(k.float(), -1).bfloat16().transpose(1, 2)
    k = k.transpose(1, 2)
    v = rn(B, L, K, V).bfloat16().transpose(1, 2)
    seg = torch.cumsum((torch.rand(B, L, device="cuda", generator=g) < 1 / 3000).long(), 1)
    dout = rn(B, L, H, V).bfloat16().transpose(1, 2)
    fwd = lambda: cfwd.adaptive_working_memory_fwd(q, qk, k, v, C, None, None, None, None, None, seg, None, return_workspace=True)
    ws = fwd()[4]
    def bwd():
        state, _ = cbwd.awm_bwd_prepare(q, qk, k, v, C, seg, workspace=ws)
        return cbwd.awm_bwd_finish(state, dout)
    for role, fn in (("fwd", fwd), ("bwd", bwd)):
        tot, top = prof(fn)
        print(f"{name:5} {role}: {tot:7.0f} us  | " + "; ".join(f"{n.strip()} {t:.0f}" for n, t in top), flush=True)
