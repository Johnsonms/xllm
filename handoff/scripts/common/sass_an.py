import csv, sys, collections
rows = list(csv.reader(open(sys.argv[1])))
hdr = rows[1]; data = rows[2:]
iA, iS, iE, iSmp = hdr.index("Address"), hdr.index("Source"), hdr.index("Instructions Executed"), hdr.index("Warp Stall Sampling (All Samples)")
num = lambda v: int(v.replace(",", "")) if v not in ("", "-") else 0
ins = [(r[iA], r[iS].strip(), num(r[iE]), num(r[iSmp])) for r in data if len(r) > iSmp]
tot = sum(e for _, _, e, _ in ins); totS = sum(s for *_, s in ins)
op = collections.Counter()
for _, src, e, _ in ins:
    o = src.split()[0] if not src.startswith("@") else src.split()[1]
    op[o.split(".")[0]] += e
print(f"total warp instr {tot:,}  samples {totS:,}")
print("top opcodes:", ", ".join(f"{k} {100*v/tot:.1f}%" for k, v in op.most_common(18)))
# windows of 40 instructions: where are executions / stall samples concentrated
W = 40
win = [(i, sum(x[2] for x in ins[i:i+W]), sum(x[3] for x in ins[i:i+W])) for i in range(0, len(ins), W)]
print("\nhottest 40-instr windows by executions (start idx, %exec, %samples):")
for i, e, s in sorted(win, key=lambda t: -t[1])[:8]:
    print(f"  {i:5d}  {100*e/tot:5.1f}%  {100*s/totS:5.1f}%   e.g. {ins[i][1][:60]}")
print("\nhottest windows by stall samples:")
for i, e, s in sorted(win, key=lambda t: -t[2])[:8]:
    print(f"  {i:5d}  {100*e/tot:5.1f}%  {100*s/totS:5.1f}%   e.g. {ins[i][1][:60]}")
