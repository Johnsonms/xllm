"""Per-stream timeline analysis of a chrome trace: compute-stream idle gaps and what ran during them.
usage: python gaps.py trace.json [top=15]"""
import json, sys, re
from collections import defaultdict
tr = json.load(open(sys.argv[1]))["traceEvents"]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 15
k = [e for e in tr if e.get("cat") == "kernel"]
# memcpy/memset also occupy streams
k += [e for e in tr if e.get("cat") in ("gpu_memcpy", "gpu_memset")]
win = [e for e in tr if e.get("cat") == "user_annotation" and e["name"] == "FWDBWD"]
if win:
    w0, w1 = win[0]["ts"], win[0]["ts"] + win[0]["dur"]
    k = [e for e in k if e["ts"] >= w0 and e["ts"] + e["dur"] <= w1]
    print(f"restricted to FWDBWD window {(w1-w0)/1000:.0f} ms")
streams = defaultdict(list)
for e in k:
    streams[e["args"]["stream"]].append(e)
def isnccl(e): return "nccl" in e["name"].lower()
print("streams:")
for s, evs in sorted(streams.items(), key=lambda x: -sum(e["dur"] for e in x[1])):
    tot = sum(e["dur"] for e in evs) / 1000
    nn = sum(e["dur"] for e in evs if isnccl(e)) / 1000
    print(f"  stream {s}: {len(evs)} kernels, {tot:.0f} ms (nccl {nn:.0f} ms), first {evs[0]['name'][:60]}")
t0 = min(e["ts"] for e in k); t1 = max(e["ts"] + e["dur"] for e in k)
print(f"span {(t1-t0)/1000:.0f} ms")
def union(evs):
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in evs); out = []
    for a, b in iv:
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out
comp = [e for e in k if not isnccl(e)]
nccl = [e for e in k if isnccl(e)]
cu = union(comp); nu = union(nccl)
cls = defaultdict(float)
for e in comp:
    n = e["name"]
    c = "gemm" if re.search("gemm|nvjet|cutlass", n) else ("copy" if re.search("copy|Copy|Cat", n) else ("sca" if "Sca" in n else ("awm" if "Awm" in n else ("tsdn" if re.search("Tsdn|Timestep|timestep", n) else "other"))))
    cls[c] += e["dur"] / 1000
print("compute by class (ms): " + ", ".join(f"{k}={v:.0f}" for k, v in sorted(cls.items(), key=lambda x: -x[1])))
print(f"compute busy {sum(b-a for a,b in cu)/1000:.0f} ms, nccl busy {sum(b-a for a,b in nu)/1000:.0f} ms")
# gaps in compute
gaps = [(cu[i][1], cu[i+1][0]) for i in range(len(cu)-1)]
gaps = [(a, b) for a, b in gaps if b - a > 20]
tot = sum(b - a for a, b in gaps)
# during each gap: was nccl running?
def overl(a, b, iv):
    return sum(max(0, min(b, y) - max(a, x)) for x, y in iv)
with_nccl = sum(overl(a, b, nu) for a, b in gaps)
print(f"kernels in window: {len(comp)} compute, {len(nccl)} nccl")
print(f"compute gaps >20us: {len(gaps)}, total {tot/1000:.0f} ms, of which nccl running {with_nccl/1000:.0f} ms")
# which nccl kernel ran during the gap and which compute kernel came after
by_end = sorted(comp, key=lambda e: e["ts"] + e["dur"])
by_start = sorted(comp, key=lambda e: e["ts"])
import bisect
starts = [e["ts"] for e in by_start]
print("largest gaps (ms, nccl-covered ms, next compute kernel, nccl kernels overlapping):")
for a, b in sorted(gaps, key=lambda g: -(g[1]-g[0]))[:top]:
    i = bisect.bisect_left(starts, b); nxt = by_start[i]["name"][:50] if i < len(by_start) else "-"
    ov = [e["name"][:40] + f"({e['dur']/1000:.1f})" for e in nccl if e["ts"] < b and e["ts"] + e["dur"] > a]
    print(f"  t={(a-t0)/1000:7.1f} {(b-a)/1000:6.2f} {overl(a,b,nu)/1000:6.2f}  next={nxt}  nccl={ov[:3]}")
# histogram of gap by phase (time position)
import math
nb = 10
hist = [0.0]*nb
for a, b in gaps:
    hist[min(nb-1, int((a - t0) / (t1 - t0) * nb))] += (b - a)/1000
print("gap ms per 10% of the step:", [f"{h:.0f}" for h in hist])
# nccl kernel name breakdown
nb2 = defaultdict(float)
for e in nccl: nb2[re.sub(r"_[0-9a-f]+$", "", e["name"])[:60]] += e["dur"]/1000
for n, v in sorted(nb2.items(), key=lambda x: -x[1])[:8]: print(f"  nccl {v:7.0f} ms  {n}")
