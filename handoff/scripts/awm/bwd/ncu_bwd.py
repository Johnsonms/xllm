import importlib.util, sys
sys.path.insert(0, "/home/johnson/xllm/tests")
import torch
from test_cute_adaptive_working_memory import awm, make_inputs
_s = importlib.util.spec_from_file_location("cute_awm_bwd", "/home/johnson/xllm/xllm/modules/fused_ops/cute/adaptive_working_memory_bwd.py")
bwd = importlib.util.module_from_spec(_s); _s.loader.exec_module(bwd)
B, H, K, S, V, L, C = (int(x) for x in sys.argv[1].split(","))
q, qk, k, v, _, _, _, _, _, seg, _ = make_inputs(B, H, K, S, V, L, C, False, 0.3)
dout = torch.randn(B, L, H, V, device="cuda").to(torch.bfloat16).transpose(1, 2)
for _ in range(2):
    bwd.adaptive_working_memory_bwd(dout, None, None, q, qk, k, v, C, seg)
torch.cuda.synchronize()
