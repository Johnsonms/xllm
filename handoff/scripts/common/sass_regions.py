"""Split the SASS into regions at BAR / SYNCS / WARPSYNC-ish markers and report execution share."""
import csv, sys
rows = list(csv.reader(open(sys.argv[1]))); hdr = rows[1]; data = rows[2:]
iS, iE, iSmp = hdr.index("Source"), hdr.index("Instructions Executed"), hdr.index("Warp Stall Sampling (All Samples)")
num = lambda v: int(v.replace(",", "")) if v not in ("", "-") else 0
ins = [(r[iS].strip(), num(r[iE]), num(r[iSmp])) for r in data if len(r) > iSmp]
tot = sum(e for _, e, _ in ins)
regs, start = [], 0
for i, (src, e, _) in enumerate(ins):
    if any(k in src for k in ("BAR.SYNC", "BAR.ARV", "SYNCS.", "UTMALDG", "UTMASTG", "RED.", "ATOM")) or i == len(ins) - 1:
        regs.append((start, i)); start = i + 1
out = []
for a, b in regs:
    e = sum(x[1] for x in ins[a:b + 1]); smp = sum(x[2] for x in ins[a:b + 1])
    out.append((e, smp, a, b, ins[b][0][:50]))
for e, smp, a, b, end in sorted(out, reverse=True)[:12]:
    ops = {}
    for src, ex, _ in ins[a:b + 1]:
        o = src.split()[1] if src.startswith("@") else src.split()[0]
        ops[o.split(".")[0]] = ops.get(o.split(".")[0], 0) + ex
    top = ", ".join(f"{k}{100*v/max(e,1):.0f}%" for k, v in sorted(ops.items(), key=lambda t: -t[1])[:6])
    print(f"[{a:5d}-{b:5d}] {100*e/tot:5.1f}% exec  {smp:5d} smp  len {b-a+1:4d}  ends '{end}'  | {top}")
