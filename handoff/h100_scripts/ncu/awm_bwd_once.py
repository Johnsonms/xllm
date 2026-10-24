"""Run the cute AWM backward of one shape once (for ncu). usage: python awm_bwd_once.py 0.9B|4B|7.1B"""
import sys, types, importlib
import torch
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
cfwd = importlib.import_module("xllm.modules.fused_ops.cute.adaptive_working_memory")
cbwd = importlib.import_module("xllm.modules.fused_ops.cute.adaptive_working_memory_bwd")
SH = {"0.9B": (2, 8, 4, 256, 256), "4B": (1, 32, 8, 128, 128), "7.1B": (1, 4, 4, 256, 2048)}
B, H, K, S, V = SH[sys.argv[1]]
L, C = 16384, 4096
g = torch.Generator(device="cuda").manual_seed(0)
rn = lambda *s: torch.randn(*s, device="cuda", generator=g)
q = torch.softmax(rn(B, L, H, S) * 2, -1).bfloat16().transpose(1, 2)
k = (rn(B, L, K, S) * 2).bfloat16()
qk = torch.softmax(k.float(), -1).bfloat16().transpose(1, 2)
k = k.transpose(1, 2)
v = rn(B, L, K, V).bfloat16().transpose(1, 2)
seg = torch.cumsum((torch.rand(B, L, device="cuda", generator=g) < 1 / 3000).long(), 1)
dout = rn(B, L, H, V).bfloat16().transpose(1, 2)
ws = cfwd.adaptive_working_memory_fwd(q, qk, k, v, C, None, None, None, None, None, seg, None, return_workspace=True)[4]
for _ in range(2):
    state, _ = cbwd.awm_bwd_prepare(q, qk, k, v, C, seg, workspace=ws)
    cbwd.awm_bwd_finish(state, dout)
torch.cuda.synchronize()
