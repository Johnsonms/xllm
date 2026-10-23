"""Backward GPU time per call (events, back-to-back) and the per-phase split."""
import importlib.util, sys
sys.path.insert(0, "/home/johnson/xllm/tests")
import torch
from test_cute_adaptive_working_memory import awm, make_inputs
_s = importlib.util.spec_from_file_location("cute_awm_bwd", "/home/johnson/xllm/xllm/modules/fused_ops/cute/adaptive_working_memory_bwd.py")
bwd = importlib.util.module_from_spec(_s); _s.loader.exec_module(bwd)

def timeit(fn, n=20, reps=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(n):
            fn()
        e1.record(); e1.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1000 / n)
    return best

for a in sys.argv[1:]:
    B, H, K, S, V, L, C = (int(x) for x in a.split(","))
    q, qk, k, v, _, _, _, _, _, seg, _ = make_inputs(B, H, K, S, V, L, C, False, 0.3)
    dout = torch.randn(B, L, H, V, device="cuda").to(torch.bfloat16).transpose(1, 2)
    full = timeit(lambda: bwd.adaptive_working_memory_bwd(dout, None, None, q, qk, k, v, C, seg))
    w = awm._phases12(qk, k, v, seg, None, None, None, None, None, None, C, save=True)
    t12 = timeit(lambda: awm._phases12(qk, k, v, seg, None, None, None, None, None, None, C, save=True))
    qb, qkb, kb, vb, db = awm._blhs(q), awm._blhs(qk), awm._blhs(k), awm._blhs(v), awm._blhs(dout)
    t3 = timeit(lambda: bwd._phase3_bwd(db, qb, w, H, K, S, V, C, False, True))
    dq, out_mask, dMo = bwd._phase3_bwd(db, qb, w, H, K, S, V, C, False, True)
    t2 = timeit(lambda: bwd._phase2_bwd(w, dMo, None, None, K, S, V, C, False, True))
    dB16, dA16, delta, dlse, _, _ = bwd._phase2_bwd(w, dMo, None, None, K, S, V, C, False, True)
    t1 = timeit(lambda: bwd._phase1_bwd(w, dA16, dB16, delta, dlse, qkb, kb, vb, K, S, V, C, True))
    print(f"B{B} H{H} K{K} S{S} V{V}: bwd {full:.0f} us/call = fwd phases 1+2 {t12:.0f} + phase 3' {t3:.0f} + phase 2' {t2:.0f} + phase 1' {t1:.0f}")
