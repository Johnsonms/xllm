import importlib.util, torch
_s = importlib.util.spec_from_file_location("cute_conv", "/home/johnson/xllm/xllm/modules/fused_ops/cute/causal_conv.py")
cv = importlib.util.module_from_spec(_s); _s.loader.exec_module(cv)
torch.manual_seed(0)
B, L, D = 2, 16384, 2048
x = torch.randn(B, L, D, device="cuda").to(torch.bfloat16); w = torch.softmax(torch.randn(D, 4, device="cuda"), -1)
bos = torch.rand(B, L, device="cuda") < 0.001; g = torch.randn_like(x)
for _ in range(2):
    o = cv.conv_fwd(x, w, bos, "silu"); dx, dw = cv.conv_bwd(g, x, w, bos, "silu")
torch.cuda.synchronize()
