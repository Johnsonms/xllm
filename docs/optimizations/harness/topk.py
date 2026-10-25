"""Top kernels by total time in the profiled step (kernel-only trace)."""
import json, sys, re
from collections import defaultdict
tr = json.load(open(sys.argv[1]))["traceEvents"]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 40
k = [e for e in tr if e.get("cat") == "kernel"]
tot = sum(e["dur"] for e in k) / 1000
agg = defaultdict(lambda: [0.0, 0])
for e in k:
    n = re.sub(r"<.*", "", e["name"])[:95]
    agg[n][0] += e["dur"] / 1000; agg[n][1] += 1
print(f"total kernel time {tot:.0f} ms, {len(k)} kernels")
for n, (t, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:top]:
    print(f"{t:7.1f} ms {100*t/tot:5.1f}% x{c:5d} {1000*t/c:7.0f} us  {n}")
