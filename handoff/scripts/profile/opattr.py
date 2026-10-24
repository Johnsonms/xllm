"""Attribute kernel time to the innermost aten op (with input dims) under the innermost xllm python frame."""
import json, re, sys
from collections import defaultdict
trace, pat = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else ".")
tr = json.load(open(trace))["traceEvents"]
frames = defaultdict(list); ops = defaultdict(list)
for e in tr:
    c = e.get("cat")
    if c == "python_function" and ("/xllm/xllm/" in e["name"] or e["name"].startswith("xllm/")):
        frames[e["tid"]].append((e["ts"], e["ts"] + e["dur"], re.sub(r"^.*?/xllm/xllm/", "xllm/", e["name"])))
    elif c == "cpu_op" and e["name"].startswith("aten::"):
        dims = e.get("args", {}).get("Input Dims", "")
        ops[e["tid"]].append((e["ts"], e["ts"] + e["dur"], e["name"] + " " + str(dims)[:70]))
for t in frames: frames[t].sort()
for t in ops: ops[t].sort()
launch = {e["args"]["correlation"]: (e["ts"], e["tid"]) for e in tr if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
kern = sorted([(launch.get(e["args"].get("correlation"), (-1, 0)), e["dur"], e["name"]) for e in tr if e.get("cat") == "kernel"])
def innermost(lst, ts):
    best = None
    for a, b, n in lst:
        if a <= ts <= b: best = n
        if a > ts: break
    return best
tot = defaultdict(float); cnt = defaultdict(int); alltot = 0
for (ts, tid), dur, name in kern:
    alltot += dur
    if not re.search(pat, name): continue
    fr = innermost(frames[tid], ts) or "?"
    op = innermost(ops[tid], ts) or "?"
    key = f"{fr} | {op}"
    tot[key] += dur / 1000; cnt[key] += 1
print(f"kernels matching '{pat}': {sum(tot.values()):.0f} ms of {alltot/1000:.0f} ms")
for s, t in sorted(tot.items(), key=lambda x: -x[1])[:60]:
    print(f"{t:7.1f} ms x{cnt[s]:4d}  {s}")
