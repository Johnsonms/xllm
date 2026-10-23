"""Instrument the backward with per-tile globaltimer events (TSDN_PROF=1). Scratch-only patch."""
import sys
p = sys.argv[1]
s = open(p).read()


def sub(old, new, cnt=1, start=0):
    global s
    head, tail = s[:start], s[start:]
    if tail.count(old) == 0:  # variants without empty_bar / with per-warp sIn slots
        for a, b in ((", empty_bar)", ")"), ("sIn[jt % nb, None, None], sInC[jt % nb, None]", "sIn[rw, None, None], sInC[rw, None]")):
            old, new = old.replace(a, b), new.replace(a, b)
    assert tail.count(old) == cnt, (tail.count(old), old[:120])
    s = head + tail.replace(old, new)


s = s.replace("\nimport math\n", "\nimport math\nimport os\n_PROF = os.environ.get('TSDN_PROF') == '1'\nNEV = 8\n", 1)
bwd = s.index("class TimestepDecayNormBwd")
sub('''        mIncW = cute.make_tensor(pWs64 + (2 + num_tiles * self.Gt * 2), cute.make_layout(num_tiles * self.Gt * 2))
''', '''        mIncW = cute.make_tensor(pWs64 + (2 + num_tiles * self.Gt * 2), cute.make_layout(num_tiles * self.Gt * 2))
        mProf = cute.make_tensor(pWs64 + (2 + num_tiles * self.Gt * 4 + self.H), cute.make_layout(num_tiles * NEV + 1))
''', start=bwd)


def ev(e, tk_expr, cond="lane == 0"):
    return "if const_expr(_PROF):\n    if %s:\n        mProf[(%s) * NEV + %d] = cute.arch.globaltimer()\n" % (cond, tk_expr, e)


def ind(code, n):
    return "".join(" " * n + l + "\n" for l in code.splitlines())


bwd = s.index("class TimestepDecayNormBwd")
# compute warps (warp 0 lane 0): E0 row sums start, E5 finish start, E6 finish end
sub('''                else:
                    buf = it % nb
                    self._row_sums(''', '''                else:
                    buf = it % nb
''' + ind(ev(0, "cur", "tidx == 0"), 20) + '''                    self._row_sums(''', start=bwd)
sub('''                    old = rOld[0]
                    if old >= 0:
                        self._finish_bwd(sTile, it - self.D,''', '''                    old = rOld[0]
                    if old >= 0:
''' + ind(ev(5, "old", "tidx == 0"), 24) + '''                        self._finish_bwd(sTile, it - self.D,''', start=bwd)
sub('''                                         beta1, beta2, log2_beta1, log2_beta2, empty_bar)
                    for d in cutlass.range_constexpr(self.D - 1):''', '''                                         beta1, beta2, log2_beta1, log2_beta2)
''' + ind(ev(6, "old", "tidx == 0"), 24) + '''                    for d in cutlass.range_constexpr(self.D - 1):''', start=bwd)
# aggregate warp: E1 start, E2 published
sub('''                else:
                    self._aggregate_bwd(''', '''                else:
''' + ind(ev(1, "tk"), 20) + '''                    self._aggregate_bwd(''', start=bwd)
sub('''                                        sAgg, sAggMaskI, beta1, beta2)
''', '''                                        sAgg, sAggMaskI, beta1, beta2)
''' + ind(ev(2, "tk"), 20), start=bwd)
# resolve: E3 start, E4 end
sub('''                else:
                    self._resolve_bwd(''', '''                else:
''' + ind(ev(3, "tk"), 20) + '''                    self._resolve_bwd(''', start=bwd)
sub('''                                      sAgg, sAggMaskI, sIn[jt % nb, None, None], sInC[jt % nb, None], log2_beta1, log2_beta2)
''', '''                                      sAgg, sAggMaskI, sIn[jt % nb, None, None], sInC[jt % nb, None], log2_beta1, log2_beta2)
''' + ind(ev(4, "tk"), 20), start=bwd)
# producer: E7 TMA issue (and the CTA in the final slot word's low bits is not needed: block from E7 order)
sub('''                            if lane == 0:
                                cute.arch.mbarrier_arrive_and_expect_tx(full_bar + stage, 2 * self.T * self.W * self.dtype.width // 8)
''', '''                            if lane == 0:
                                cute.arch.mbarrier_arrive_and_expect_tx(full_bar + stage, 2 * self.T * self.W * self.dtype.width // 8)
''' + ind(ev(7, "tk"), 28), start=bwd)
open(p, "w").write(s)
s = open(p).read()
old = "    ws = torch.zeros(ws_words + H, dtype=torch.int64, device=x.device)\n"
assert s.count(old) == 1
s = s.replace(old, "    ws = torch.zeros(ws_words + H + (num_tiles * NEV + 1 if _PROF else 0), dtype=torch.int64, device=x.device)\n    global _LAST_PROF\n    _LAST_PROF = (ws[ws_words + H:], num_tiles, num_chains, num_segs)\n")
open(p, "w").write(s)
