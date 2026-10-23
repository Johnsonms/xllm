import sys
import torch
from torch.profiler import profile, ProfilerActivity
from check_awm_fwd import make, as_blhs_views, cute_awm

for a in sys.argv[1:]:
    B, H, K, S, V, L, C = (int(x) for x in a.split(","))
    args = list(make(B, H, K, S, V, L, C, False, True, dtype=torch.bfloat16))
    args[:4] = as_blhs_views(*args[:4])
    q, qk, k, v, memory, lnt, pqk, pk, pv, seg, cnt = args
    fn = lambda: cute_awm.adaptive_working_memory_fwd(q, qk, k, v, C, None, None, None, None, None, seg, None)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
    print(f"== B{B} H{H} K{K} S{S} V{V} L{L} C{C}")
    rows = sorted(prof.key_averages(), key=lambda e: -e.device_time_total)
    for e in rows[:12]:
        print(f"  {e.device_time_total / 10:8.1f} us  x{e.count // 10:<3d} {e.key[:90]}")
