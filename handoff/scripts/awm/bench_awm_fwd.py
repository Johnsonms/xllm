import sys
import time
import torch
from bench_kernel import gpu_time
from check_awm_fwd import make, as_blhs_views, ref_mod, cute_awm

for a in sys.argv[1:]:
    B, H, K, S, V, L, C = (int(x) for x in a.split(","))
    args = list(make(B, H, K, S, V, L, C, False, True, dtype=torch.bfloat16))
    args[:4] = as_blhs_views(*args[:4])
    q, qk, k, v, memory, lnt, pqk, pk, pv, seg, cnt = args
    call_ref = lambda: ref_mod._adaptive_working_memory_fwd(q, qk, k, v, C, None, None, None, None, None, seg, None)
    call_cute = lambda: cute_awm.adaptive_working_memory_fwd(q, qk, k, v, C, None, None, None, None, None, seg, None)
    res = []
    for name, fn in (("ref", call_ref), ("cute", call_cute)):
        g = gpu_time(fn)
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        res.append((name, g, (time.time() - t0) / 10 * 1e6))
    print(f"B{B} H{H} K{K} S{S} V{V} L{L} C{C}: " + "  ".join(f"{n}: GPU {g:7.1f} us, wall {w:7.1f} us" for n, g, w in res) +
          f"  speedup GPU {res[0][1] / res[1][1]:.1f}x wall {res[0][2] / res[1][2]:.1f}x")
