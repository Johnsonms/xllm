import json, re, sys
from collections import defaultdict
res = {}
for t in sys.argv[1:]:
    tr = json.load(open(t))["traceEvents"]
    win = [e for e in tr if e.get("cat") == "user_annotation" and e["name"] == "FWDBWD"][0]
    w0, w1 = win["ts"], win["ts"] + win["dur"]
    k = [e for e in tr if e.get("cat") == "kernel" and w0 <= e["ts"] <= w1]
    agg = defaultdict(lambda: [0, 0.0])
    for e in k:
        n = re.sub(r"<.*", "", e["name"]); n = re.sub(r"kernel_cutlass_kernel_xllmmodulesfused_opscute", "cute:", n)[:80]
        agg[n][0] += 1; agg[n][1] += e["dur"] / 1000
    res[t] = (agg, (w1 - w0) / 1000, sum(e["dur"] for e in k) / 1000)
names = set().union(*[r[0].keys() for r in res.values()])
print("window ms:", {t: f"{r[1]:.0f}" for t, r in res.items()}, "kernel sum ms:", {t: f"{r[2]:.0f}" for t, r in res.items()})
rows = sorted(names, key=lambda n: -abs(res[sys.argv[1]][0][n][1] - res[sys.argv[2]][0][n][1]))
print(f"{'A ms':>8} {'B ms':>8} {'diff':>7} {'nA':>5} {'nB':>5}  kernel")
for n in rows[:16]:
    a, b = res[sys.argv[1]][0][n], res[sys.argv[2]][0][n]
    print(f"{a[1]:8.1f} {b[1]:8.1f} {b[1]-a[1]:7.1f} {a[0]:5d} {b[0]:5d}  {n}")
