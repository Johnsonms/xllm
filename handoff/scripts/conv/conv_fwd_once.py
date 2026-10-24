import importlib.util, torch
_s = importlib.util.spec_from_file_location("cute_conv", "/tmp/claude-243000396/-data-home-johnson-xllm/d3839bfb-7816-4668-9945-7c337f0c3f9e/scratchpad/sca/causal_conv_draft.py")
cv = importlib.util.module_from_spec(_s); _s.loader.exec_module(cv)
torch.manual_seed(0)
B, L, D = 2, 16384, 2048
x = torch.randn(B, L, D, device="cuda").to(torch.bfloat16); w = torch.softmax(torch.randn(D, 4, device="cuda"), -1)
bos = torch.rand(B, L, device="cuda") < 0.001
for _ in range(3): o = cv.conv_fwd(x, w, bos, None)
torch.cuda.synchronize()
