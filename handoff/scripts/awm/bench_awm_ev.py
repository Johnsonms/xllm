import sys
import torch
from check_awm_fwd import make, as_blhs_views, cute_awm

for a in sys.argv[1:]:
    B, H, K, S, V, L, C = (int(x) for x in a.split(","))
    args = list(make(B, H, K, S, V, L, C, False, True, dtype=torch.bfloat16))
    args[:4] = as_blhs_views(*args[:4])
    q, qk, k, v, memory, lnt, pqk, pk, pv, seg, cnt = args
    fn = lambda: cute_awm.adaptive_working_memory_fwd(q, qk, k, v, C, None, None, None, None, None, seg, None)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(5):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(20):
            fn()
        e1.record()
        e1.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1000 / 20)
    print(f"B{B} H{H} K{K} S{S} V{V}: {best:.1f} us per call (events, 20 back-to-back)")
