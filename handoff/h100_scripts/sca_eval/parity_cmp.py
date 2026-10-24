import re,sys
def read(f):
    rows = {}
    for l in open(f):
        m = re.match(r"step (\d+): loss ([\d.]+) gnorm ([\d.]+)\s+fwd\+bwd (\d+) ms", l)
        if m: rows[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), int(m.group(4)))
    return rows
o, r = read(sys.argv[1]), read(sys.argv[2])
common = sorted(set(o) & set(r))
dl = [abs(o[s][0] - r[s][0]) for s in common]; dg = [abs(o[s][1] - r[s][1]) / r[s][1] for s in common]
print(f"steps {len(common)}; loss |diff| max {max(dl):.4f} mean {sum(dl)/len(dl):.5f}; gnorm rel diff max {max(dg):.4f}")
for s in (0, 20, 40, len(common) - 1):
    if s in o and s in r: print(f"  step {s:2d}: opt {o[s][0]:.4f}/{o[s][1]:.4f} | ref {r[s][0]:.4f}/{r[s][1]:.4f}")
med = lambda d: sorted(v[2] for k, v in d.items() if k >= 5)[len([k for k in d if k >= 5]) // 2]
print(f"median fwd+bwd (steps 5+): opt {med(o)} ms, ref {med(r)} ms")
