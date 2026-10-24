"""Kernel time of one profiled step grouped by Gekko op, using the python call stack of each launch (needs WITH_STACK=1).

usage: python cats.py trace.json [trace2.json ...]
Categories are matched on the enclosing xllm frames (innermost first), so e.g. the GEMMs inside the SCA wrappers count as SCA.
"""
import json, re, sys
from collections import defaultdict

CATS = [  # (category, regex on a frame "file(line): func"), first match innermost-out wins
    ("sca", r"attention/sca\.py|sliding_chunk_attention"),
    ("tsdn", r"timenorm|timestep_decay_norm"),
    ("awm", r"adaptive_working_memory|awm_"),
    ("conv", r"causal_conv|conv1d"),
    ("ffn", r"swiglu"),
    ("ce", r"cross_entropy"),
    ("rope", r"rope"),
    ("norm", r"rmsnorm|rms_norm|layer_norm|group_norm"),
    ("optim", r"optim|adam|clip_grad"),
]


def analyze(path):
    tr = json.load(open(path))["traceEvents"]
    frames = defaultdict(list)
    for e in tr:
        if e.get("cat") == "python_function" and (e["name"].startswith("xllm/") or "/xllm/xllm/" in e["name"]):
            frames[e["tid"]].append((e["ts"], e["ts"] + e["dur"], re.sub(r"^.*?/xllm/xllm/", "xllm/", e["name"])))
    for t in frames:
        frames[t].sort()
    launch = {e["args"]["correlation"]: (e["ts"], e["tid"]) for e in tr
              if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    kern = sorted([(launch.get(e["args"].get("correlation"), (-1, 0)), e["dur"], e["name"]) for e in tr if e.get("cat") == "kernel"])
    tot = defaultdict(float)
    fi, stack = defaultdict(int), defaultdict(list)
    for (ts, tid), dur, name in kern:
        fr, st = frames[tid], stack[tid]
        while fi[tid] < len(fr) and fr[fi[tid]][0] <= ts:
            while st and st[-1][1] <= fr[fi[tid]][0]:
                st.pop()
            st.append(fr[fi[tid]])
            fi[tid] += 1
        while st and st[-1][1] < ts:
            st.pop()
        if "nccl" in name.lower():
            cat = "nccl"
        elif re.search(r"multi_tensor_apply|adam", name, re.I):
            cat = "optim"
        else:
            cat = "other"
            for f in reversed(st):
                hit = next((c for c, rx in CATS if re.search(rx, f[2])), None)
                if hit:
                    cat = hit
                    break
            if cat == "other" and re.search(r"gemm|nvjet|xmma|cutlass", name):
                cat = "gemm(other)"
        tot[cat] += dur / 1000
    return tot


if __name__ == "__main__":
    for p in sys.argv[1:]:
        tot = analyze(p)
        comp = sum(v for k, v in tot.items() if k != "nccl")
        print(f"== {p}: non-nccl kernel time {comp:.0f} ms" + (f", nccl {tot['nccl']:.0f} ms" if "nccl" in tot else ""))
        print("   " + ", ".join(f"{k} {v:.0f} ({100 * v / comp:.0f}%)" for k, v in sorted(tot.items(), key=lambda x: -x[1]) if k != "nccl"))
