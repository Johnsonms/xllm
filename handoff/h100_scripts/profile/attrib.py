import json, re, sys
from collections import defaultdict
trace, pat = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else ".")
depth = int(sys.argv[3]) if len(sys.argv) > 3 else 3
tr = json.load(open(trace))["traceEvents"]
frames = defaultdict(list)
for e in tr:
    if e.get("cat") == "python_function" and (e["name"].startswith("xllm/") or "/xllm/xllm/" in e["name"]):
        frames[e["tid"]].append((e["ts"], e["ts"] + e["dur"], re.sub(r"^.*?/xllm/xllm/", "xllm/", e["name"])))
for t in frames: frames[t].sort()
launch = {e["args"]["correlation"]: (e["ts"], e["tid"]) for e in tr if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
kern = sorted([(launch.get(e["args"].get("correlation"), (-1, 0)), e["dur"], e["name"]) for e in tr if e.get("cat") == "kernel"])
tot = defaultdict(float); kn = defaultdict(lambda: defaultdict(float)); alltot = 0.0
fi = defaultdict(int); stack = defaultdict(list)
for (ts, tid), dur, name in kern:
    alltot += dur
    fr, st = frames[tid], stack[tid]
    while fi[tid] < len(fr) and fr[fi[tid]][0] <= ts:
        while st and st[-1][1] <= fr[fi[tid]][0]: st.pop()
        st.append(fr[fi[tid]]); fi[tid] += 1
    while st and st[-1][1] < ts: st.pop()
    if not re.search(pat, name): continue
    key = " <- ".join(f[2] for f in reversed(st[-depth:])) if st else f"?(tid {tid})"
    tot[key] += dur / 1000; kn[key][name[:90]] += dur / 1000
print(f"kernels matching '{pat}': {sum(tot.values()):.0f} ms of {alltot/1000:.0f} ms")
for s, t in sorted(tot.items(), key=lambda x: -x[1])[:40]:
    print(f"{t:7.1f} ms  {s}")
    for n, tt in sorted(kn[s].items(), key=lambda x: -x[1])[:3]:
        print(f"           {tt:6.1f}  {n}")
