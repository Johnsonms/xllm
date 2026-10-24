"""Attribute GPU-idle time (compute-stream gaps inside the FWDBWD window) to the python frames active on the main thread.
usage: python gapattr.py trace.json [depth=2] [filter regex for frame names, default xllm|fsdp]"""
import json, sys, re, bisect
from collections import defaultdict
tr = json.load(open(sys.argv[1]))["traceEvents"]
depth = int(sys.argv[2]) if len(sys.argv) > 2 else 2
pat = sys.argv[3] if len(sys.argv) > 3 else r"xllm/|fsdp/"
win = [e for e in tr if e.get("cat") == "user_annotation" and e["name"] == "FWDBWD"][0]
w0, w1, tid = win["ts"], win["ts"] + win["dur"], win["tid"]
k = [e for e in tr if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "nccl" not in e["name"].lower() and w0 <= e["ts"] and e["ts"] + e["dur"] <= w1]
iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in k); cu = []
for a, b in iv:
    if cu and a <= cu[-1][1]: cu[-1][1] = max(cu[-1][1], b)
    else: cu.append([a, b])
gaps = [(cu[i][1], cu[i + 1][0]) for i in range(len(cu) - 1) if cu[i + 1][0] - cu[i][1] > 5]
print(f"window {(w1-w0)/1000:.0f} ms, compute busy {sum(b-a for a,b in cu)/1000:.0f} ms, gaps {sum(b-a for a,b in gaps)/1000:.0f} ms in {len(gaps)} gaps")
def short(n):
    n = re.sub(r"^.*?/xllm/xllm/", "xllm/", n); n = re.sub(r"^.*?/site-packages/", "", n); return n
# python frames on the main thread (all threads actually: backward runs on the autograd thread)
fr = [(e["ts"], e["ts"] + e["dur"], short(e["name"]), e["tid"]) for e in tr if e.get("cat") == "python_function" and re.search(pat, e["name"])]
fr.sort(); starts = [f[0] for f in fr]
tot = defaultdict(float); byfile = defaultdict(float)
step = 10.0
for a, b in gaps:
    t = a + step / 2
    while t < b:
        # frames active at t: those with start <= t < end; take the innermost (latest start) few
        i = bisect.bisect_right(starts, t)
        act = [f for f in fr[max(0, i - 400):i] if f[1] > t]
        act.sort(key=lambda f: f[0])
        key = " <- ".join(f[2] for f in reversed(act[-depth:])) if act else "?"
        w = min(step, b - t)
        tot[key] += w / 1000
        byfile[re.sub(r"\(\d+\)", "", act[-1][2]) if act else "?"] += w / 1000
        t += step
print("GPU-idle ms by innermost frame:")
for s, v in sorted(tot.items(), key=lambda x: -x[1])[:30]: print(f"{v:7.1f}  {s}")
print("GPU-idle ms by innermost function (line-agnostic):")
for s, v in sorted(byfile.items(), key=lambda x: -x[1])[:25]: print(f"{v:7.1f}  {s}")
