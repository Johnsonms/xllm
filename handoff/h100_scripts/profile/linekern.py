"""Non-GEMM kernel time launched directly from a source file, by line (deepest frame of that file on the stack).
usage: python linekern.py trace.json file_substring [top]"""
import json, re, sys
from collections import defaultdict
tr = json.load(open(sys.argv[1]))["traceEvents"]; sub = sys.argv[2]; top = int(sys.argv[3]) if len(sys.argv) > 3 else 30
frames = defaultdict(list)
for e in tr:
    if e.get("cat") == "python_function":
        frames[e["tid"]].append((e["ts"], e["ts"] + e["dur"], e["name"]))
for t in frames: frames[t].sort()
launch = {e["args"]["correlation"]: (e["ts"], e["tid"]) for e in tr if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
tot = defaultdict(float); kn = defaultdict(lambda: defaultdict(float)); n = defaultdict(int)
for e in tr:
    if e.get("cat") != "kernel" or re.search("gemm|nvjet|xmma|cutlass|flash", e["name"]): continue
    ts, tid = launch.get(e["args"].get("correlation"), (-1, 0))
    st = [f for f in frames[tid] if f[0] <= ts <= f[1]]
    mine = [f for f in st if sub in f[2]]
    if not mine: continue
    deepest = max(mine, key=lambda f: f[0])
    # skip kernels whose deeper xllm frame is another xllm file (attributed there)
    deeper = [f for f in st if f[0] > deepest[0] and "/xllm/xllm/" in f[2]]
    if deeper: continue
    key = re.sub(r"^.*?/xllm/xllm/", "", deepest[2])
    tot[key] += e["dur"] / 1000; n[key] += 1; kn[key][e["name"][:60]] += e["dur"] / 1000
print(f"total {sum(tot.values()):.1f} ms")
for k, v in sorted(tot.items(), key=lambda x: -x[1])[:top]:
    print(f"{v:6.1f} ms {n[k]:4d}x  {k}   [{'; '.join(f'{a[:40]} {b:.1f}' for a, b in sorted(kn[k].items(), key=lambda x: -x[1])[:2])}]")
