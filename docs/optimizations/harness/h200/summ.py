"""Summarise repeated runs: per config the per-rep mean fwd+bwd (steps 1..n-2), the median over all pooled steady
steps, the low quartile ("fast regime": the host CPU clock of this VM switches between two speeds), peak memory.
usage: summ.py <log dir> <prefix> ; logs named <dir>/<prefix><cfg>_r<k>.log (one harness run each)"""
import glob, os, re, statistics, sys
from collections import defaultdict

logdir, pre = sys.argv[1], sys.argv[2]
runs = defaultdict(list)
for f in sorted(glob.glob(os.path.join(os.path.expanduser(logdir), f"{pre}*_r[0-9]*.log"))):
    name = re.sub(r"_r\d+\.log$", "", os.path.basename(f))[len(pre):]
    txt = open(f).read()
    steps = [float(m) for m in re.findall(r"^step \d+: .*?fwd\+bwd (\d+) ms", txt, re.M)]
    mem = re.findall(r"peak mem ([\d.]+) GiB", txt)
    other = re.findall(r"clip (\d+) ms, optim (\d+) ms;", txt)
    if len(steps) < 4:
        err = re.search(r"(OutOfMemoryError|illegal memory access|Error[^\n]{0,60})", txt)
        runs[name].append(("fail", err.group(1)[:60] if err else "?"))
        continue
    runs[name].append((steps[1:-1], float(mem[0]) if mem else 0, sum(map(int, other[0])) if other else 0))
for name, rs in runs.items():
    ok = [r for r in rs if r[0] != "fail"]
    if not ok:
        print(f"{name:22s} FAIL {rs[0][1]}"); continue
    pooled = sorted(s for r in ok for s in r[0])
    means = [sum(r[0]) / len(r[0]) for r in ok]
    q1 = pooled[len(pooled) // 4]
    print(f"{name:22s} reps {' / '.join(f'{m:.0f}' for m in means):20s} median {statistics.median(pooled):5.0f}  q1 {q1:5.0f}  "
          f"peak {max(r[1] for r in ok):5.1f} GiB  (+clip/optim {ok[0][2]} ms)" + (f"  [{len(rs) - len(ok)} failed]" if len(ok) < len(rs) else ""))
