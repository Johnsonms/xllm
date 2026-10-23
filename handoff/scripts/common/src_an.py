"""Aggregate ncu source-level (line info) metrics by function and by line."""
import csv, sys, re, collections
rows = list(csv.reader(open(sys.argv[1])))
src_path = sys.argv[3] if len(sys.argv) > 3 else "/home/johnson/xllm/xllm/modules/fused_ops/cute/timestep_decay_norm.py"
hi = next(i for i, r in enumerate(rows) if r and r[0] == "Line No")
hdr = rows[hi]
data = [r for r in rows[hi + 1:] if len(r) == len(hdr) and r[2] == "-"]
col = {h: i for i, h in reversed(list(enumerate(hdr)))}
num = lambda v: float(v.replace(",", "")) if v not in ("", "-") else 0.0
stalls = [h for h in hdr if h.startswith("stall_") and "Not Issued" not in h]
# function ranges from the python source
lines = open(src_path).read().split("\n")
funcs = []
for i, l in enumerate(lines, 1):
    m = re.match(r"\s*def (\w+)\(", l)
    if m:
        funcs.append((i, m.group(1)))
def func_of(n):
    name = "?"
    for start, f in funcs:
        if start <= n:
            name = f
    return name
by_f = collections.defaultdict(lambda: collections.Counter())
tot_e = tot_s = 0
per_line = []
for r in data:
    n = int(r[0]); e = num(r[col["Instructions Executed"]]); smp = num(r[col["Warp Stall Sampling (All Samples)"]])
    f = func_of(n)
    by_f[f]["exec"] += e; by_f[f]["samples"] += smp
    for st in stalls:
        by_f[f][st] += num(r[col[st]])
    tot_e += e; tot_s += smp
    per_line.append((smp, e, n, f, r[1].strip()[:70], {st: num(r[col[st]]) for st in stalls}))
print(f"total exec {tot_e:,.0f}  samples {tot_s:,.0f}")
print(f"{'function':22s} {'exec%':>6s} {'smp%':>6s}  top stalls")
for f, c in sorted(by_f.items(), key=lambda kv: -kv[1]["samples"])[:12]:
    top = sorted(((c[st], st) for st in stalls), reverse=True)[:4]
    print(f"{f:22s} {100*c['exec']/tot_e:6.1f} {100*c['samples']/tot_s:6.1f}  " + ", ".join(f"{st[6:]}={100*v/max(c['samples'],1):.0f}%" for v, st in top))
print("\nhottest lines by samples:")
for smp, e, n, f, s, st in sorted(per_line, reverse=True)[:int(sys.argv[2]) if len(sys.argv) > 2 else 15]:
    top = sorted(((v, k) for k, v in st.items()), reverse=True)[:2]
    print(f"  {smp:6.0f} smp {100*e/tot_e:5.1f}% ex  L{n:<5d} {f:18s} {s:60s} [{', '.join(f'{k[6:]}={v:.0f}' for v, k in top)}]")
