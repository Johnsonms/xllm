"""Per-source-line warp stall samples from an ncu --page source --print-source cuda,sass --csv export (all files)."""
import csv, sys, collections
rows = list(csv.reader(open(sys.argv[1])))
cur_file = None; hdr = None; col = None
per_line = []
for r in rows:
    if not r:
        continue
    if r[0] == "File Path":
        cur_file = r[1].split("/")[-1]; hdr = None; continue
    if r[0] == "Line No":
        hdr = r; col = {h: i for i, h in reversed(list(enumerate(hdr)))}; continue
    if hdr is None or len(r) != len(hdr) or r[2] != "-":
        continue
    num = lambda v: float(v.replace(",", "")) if v not in ("", "-") else 0.0
    smp = num(r[col["Warp Stall Sampling (All Samples)"]]); ex = num(r[col["Instructions Executed"]])
    stalls = {h[6:]: num(r[col[h]]) for h in hdr if h.startswith("stall_") and "Not Issued" not in h} if any(h.startswith("stall_") for h in hdr) else {}
    per_line.append((smp, ex, cur_file, int(r[0]), r[1].strip()[:80], stalls))
tot = sum(p[0] for p in per_line) or 1
print(f"total samples {tot:,.0f}; columns with stall_: {sum(1 for h in hdr if h.startswith('stall_'))}")
for smp, ex, f, n, src, st in sorted(per_line, reverse=True)[:28]:
    top = ", ".join(f"{k}={v/max(smp,1)*100:.0f}%" for v, k in sorted(((v, k) for k, v in st.items()), reverse=True)[:3])
    print(f"{100*smp/tot:5.1f}%  {f[:28]:28s}:{n:<5d} {src:60s} {top}")
