"""CPU self time per python function on the main thread inside the FWDBWD window (with_stack traces).
usage: python cpuself.py trace.json [top=40] [--fwd]  (--fwd: only until the first backward kernel launch)"""
import json, sys, re
from collections import defaultdict
tr = json.load(open(sys.argv[1]))["traceEvents"]
top = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 40
win = [e for e in tr if e.get("cat") == "user_annotation" and e["name"] == "FWDBWD"][0]
w0, w1, tid = win["ts"], win["ts"] + win["dur"], win["tid"]
if "--fwd" in sys.argv:
    bw = [e for e in tr if e.get("cat") == "cpu_op" and e["name"].startswith("autograd::engine") and e["ts"] > w0]
    w1 = min(e["ts"] for e in bw)
def short(n):
    n = re.sub(r"^.*?/xllm/xllm/", "xllm/", n); n = re.sub(r"^.*?/site-packages/", "", n); return re.sub(r"\(\d+\)", "", n)
fr = [e for e in tr if e.get("cat") == "python_function" and e["tid"] == tid and w0 <= e["ts"] and e["ts"] + e["dur"] <= w1]
fr.sort(key=lambda e: (e["ts"], -e["dur"]))
# self time = dur - direct python children (stack-based)
selfs = defaultdict(float); tots = defaultdict(float); calls = defaultdict(int)
stack = []
for e in fr:
    while stack and stack[-1]["ts"] + stack[-1]["dur"] <= e["ts"]: stack.pop()
    if stack: stack[-1]["_child"] = stack[-1].get("_child", 0) + e["dur"]
    stack.append(e)
for e in fr:
    n = short(e["name"]); selfs[n] += (e["dur"] - e.get("_child", 0)) / 1000; tots[n] += e["dur"] / 1000; calls[n] += 1
print(f"window {(w1-w0)/1000:.0f} ms, main-thread python frames {len(fr)}, self total {sum(selfs.values()):.0f} ms")
print(f"{'self ms':>8} {'total':>8} {'calls':>6}  function")
for n, v in sorted(selfs.items(), key=lambda x: -x[1])[:top]:
    print(f"{v:8.1f} {tots[n]:8.1f} {calls[n]:6d}  {n}")
