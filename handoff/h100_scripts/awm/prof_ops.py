"""Top aten ops by CUDA time in the cute AWM backward (0.9B shape)."""
import sys, types, importlib
import torch
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
from torch.profiler import profile, ProfilerActivity
cfwd = importlib.import_module("xllm.modules.fused_ops.cute.adaptive_working_memory")
cbwd = importlib.import_module("xllm.modules.fused_ops.cute.adaptive_working_memory_bwd")
B, H, K, S, V = {"0.9B": (2, 8, 4, 256, 256), "4B": (1, 32, 8, 128, 128), "7.1B": (1, 4, 4, 256, 2048)}[sys.argv[1]]
L, C = 16384, 4096
rn = lambda *s: torch.randn(*s, device="cuda")
q = torch.softmax(rn(B, L, H, S) * 2, -1).bfloat16().transpose(1, 2)
k = (rn(B, L, K, S) * 2).bfloat16(); qk = torch.softmax(k.float(), -1).bfloat16().transpose(1, 2); k = k.transpose(1, 2)
v = rn(B, L, K, V).bfloat16().transpose(1, 2)
seg = torch.cumsum((torch.rand(B, L, device="cuda") < 1 / 3000).long(), 1)
dout = rn(B, L, H, V).bfloat16().transpose(1, 2)
fwd = lambda: cfwd.adaptive_working_memory_fwd(q, qk, k, v, C, None, None, None, None, None, seg, None, return_workspace=True)
ws = fwd()[4]
def bwd():
    st, _ = cbwd.awm_bwd_prepare(q, qk, k, v, C, seg, workspace=ws)
    return cbwd.awm_bwd_finish(st, dout)
for name, fn in (("fwd", fwd), ("bwd", bwd)):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(10): fn()
        torch.cuda.synchronize()
    print(f"== {name}")
    for e in sorted([e for e in p.key_averages() if e.key.startswith("aten::") and e.device_time_total > 0], key=lambda e: -e.self_device_time_total)[:12]:
        print(f"  {e.self_device_time_total / 10:7.1f} us  {e.count // 10:3d}x  {e.key}")
