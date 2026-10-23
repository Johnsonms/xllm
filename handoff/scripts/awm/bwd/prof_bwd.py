"""Per-kernel GPU times of one CuTe AWM backward call (torch.profiler)."""
import importlib.util, sys
sys.path.insert(0, "/home/johnson/xllm/tests")
import torch
from collections import defaultdict
from test_cute_adaptive_working_memory import awm, make_inputs
_s = importlib.util.spec_from_file_location("cute_awm_bwd", "/home/johnson/xllm/xllm/modules/fused_ops/cute/adaptive_working_memory_bwd.py")
bwd = importlib.util.module_from_spec(_s); _s.loader.exec_module(bwd)
for a in sys.argv[1:]:
    B, H, K, S, V, L, C = (int(x) for x in a.split(","))
    q, qk, k, v, _, _, _, _, _, seg, _ = make_inputs(B, H, K, S, V, L, C, False, 0.3)
    dout = torch.randn(B, L, H, V, device="cuda").to(torch.bfloat16).transpose(1, 2)
    _, _, _, _, ws = awm.adaptive_working_memory_fwd(q, qk, k, v, C, None, None, None, None, None, seg, None, return_workspace=True)
    def fn():
        state, _ = bwd.awm_bwd_prepare(q, qk, k, v, C, seg, False, ws)
        return bwd.awm_bwd_finish(state, dout)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
    tot = defaultdict(float); cnt = defaultdict(int)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            tot[e.name[:60]] += e.device_time / 5; cnt[e.name[:60]] += 1
    total = sum(tot.values())
    print(f"B{B} H{H} K{K} S{S} V{V}: GPU total {total:.0f} us per call")
    for n, t in sorted(tot.items(), key=lambda x: -x[1]):
        print(f"  {t:7.1f} us  x{cnt[n] // 5:2d}  {n}")
