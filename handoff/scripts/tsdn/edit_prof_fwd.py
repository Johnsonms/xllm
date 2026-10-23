"""Instrument the forward with per-tile globaltimer events (TSDN_PROF=1). Scratch-only patch.
Events: 0 moments start, 1 aggregate start, 2 aggregate end, 3 resolve start, 4 resolve end,
5 finish call, 6 finish end, 7 TMA issue."""
import sys
p = sys.argv[1]
s = open(p).read()
if "_PROF = " not in s:
    s = s.replace("\nimport math\n", "\nimport math\nimport os\n_PROF = os.environ.get('TSDN_PROF') == '1'\nNEV = 8\n", 1)
end = s.index("class TimestepDecayNormBwd")


def sub(old, new, cnt=1):
    global s
    head, tail = s[:end], s[end:]
    assert head.count(old) == cnt, (head.count(old), old[:120])
    s = head.replace(old, new) + tail


def ev(e, tk_expr, cond="lane == 0"):
    return "if const_expr(_PROF):\n    if %s:\n        mProf[(%s) * NEV + %d] = cute.arch.globaltimer()\n" % (cond, tk_expr, e)


def ind(code, n):
    return "".join(" " * n + l + "\n" for l in code.splitlines())


sub('''        mIncW = cute.make_tensor(pWs64 + (2 + num_tiles * self.Gt * 2), cute.make_layout(num_tiles * self.Gt * 2))
''', '''        mIncW = cute.make_tensor(pWs64 + (2 + num_tiles * self.Gt * 2), cute.make_layout(num_tiles * self.Gt * 2))
        mProf = cute.make_tensor(pWs64 + (2 + num_tiles * self.Gt * 4), cute.make_layout(num_tiles * NEV + 1))
''')
sub('''                    self._moments(sX,''', ind(ev(0, "cur", "tidx == 0"), 20) + '''                    self._moments(sX,''')
sub('''                    if old >= 0:
                        self._finish_tile(''', '''                    if old >= 0:
''' + ind(ev(5, "old", "tidx == 0"), 24) + '''                        self._finish_tile(''')
sub('''                                          stages, nb, tile_elems, w_dyn, vec_dyn)
                    for d in cutlass.range_constexpr(self.D - 1):''', '''                                          stages, nb, tile_elems, w_dyn, vec_dyn)
''' + ind(ev(6, "old", "tidx == 0"), 24) + '''                    for d in cutlass.range_constexpr(self.D - 1):''')
sub('''                else:
                    self._aggregate_tile(''', '''                else:
''' + ind(ev(1, "tk"), 20) + '''                    self._aggregate_tile(''')
sub('''                                         sAgg, sAggMaskI, beta1, beta2, one_minus_beta1, one_minus_beta2)
''', '''                                         sAgg, sAggMaskI, beta1, beta2, one_minus_beta1, one_minus_beta2)
''' + ind(ev(2, "tk"), 20))
sub('''                else:
                    self._resolve_tile(''', '''                else:
''' + ind(ev(3, "tk"), 20) + '''                    self._resolve_tile(''')
sub('''                                       log2_beta1, log2_beta2, eps, rsqrt_eps)
''', '''                                       log2_beta1, log2_beta2, eps, rsqrt_eps)
''' + ind(ev(4, "tk"), 20))
i = s.index("cute.arch.mbarrier_arrive_and_expect_tx(full_bar + stage")
j = s.index("\n", i) + 1
assert i < end
s = s[:j] + ind(ev(7, "tk"), 32) + s[j:]
old = "    ws = torch.zeros(2 + num_tiles * gt * 4, dtype=torch.int64, device=x.device)\n"
assert s.count(old) == 1
s = s.replace(old, "    ws = torch.zeros(2 + num_tiles * gt * 4 + (num_tiles * NEV + 1 if _PROF else 0), dtype=torch.int64, device=x.device)\n    global _LAST_PROF_FWD\n    _LAST_PROF_FWD = (ws[2 + num_tiles * gt * 4:], num_tiles, B * (G // gt), num_segs)\n")
open(p, "w").write(s)
