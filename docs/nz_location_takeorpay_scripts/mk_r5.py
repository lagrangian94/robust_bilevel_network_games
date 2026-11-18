# -*- coding: utf-8 -*-
# builds the scripts of the radius 0.5 run (ledger 4.33) from the existing ones, without touching them:
#   dwD5W.jl  = dwLeanB.jl lines 1-117 + dwD5_body.jl
#   dwB5W.jl  = dwLeanB.jl lines 1-117 + dwB5_body.jl
#   coupW5.jl = coupWE.jl with the candidate list read from candx_<cfg>_<CP_CX>_g_s.csv (extended candidates) and an optional suffix / candidate limit for tests
import io
def rd(p): return io.open(p, encoding="utf-8").read()
def wr(p, s): io.open(p, "w", encoding="utf-8", newline="\n").write(s)
head = "".join(io.open("dwLeanB.jl", encoding="utf-8").readlines()[:117])
wr("dwD5W.jl", head + rd("dwD5_body.jl")); wr("dwB5W.jl", head + rd("dwB5_body.jl"))
s = rd("coupWE.jl")
a = 'cand = [IDX[String(split(l, ",")[1])] for l in readlines(SPD * "geo/dw/dw_$(tag0).csv")[2:end]]; nc = length(cand)'
b = ('const CX = ENV["CP_CX"]; const SUF = E("D5_SUF", ""); const CPMAX = parse(Int, E("CP_MAXC", "0"))   # coupW5.jl (2026-11-18): extended candidates of dwD5W.jl\n'
     'CL = [split(l, ",") for l in readlines(SPD * "geo/dw/candx_$(CCFG)_$(CX)_g$(GG)_s$(SEED)$(SUF).csv")[2:end]]; CPMAX > 0 && (CL = vcat(CL[1:min(CPMAX, length(CL))], [c for c in CL[min(CPMAX, length(CL))+1:end] if c[2] == "1"]))\n'
     'cand = [IDX[String(c[1])] for c in CL]; nc = length(cand); @printf("CANDX %d candidates, %d new\\n", nc, count(c -> c[2] == "1", CL))')
assert s.count(a) == 1; s = s.replace(a, b)
for old in ('"geo/dw/cpx_$(CCFG)_$(CLAB)_g$(GG)_s$(SEED).csv"', '"geo/dw/selx_$(CCFG)_$(CLAB)_g$(GG)_s$(SEED).csv"'):
    assert s.count(old) == 1, old; s = s.replace(old, old.replace('$(SEED).csv"', '$(SEED)$(SUF).csv"'))
wr("coupW5.jl", s)
print("built dwD5W.jl (%d lines), dwB5W.jl (%d lines), coupW5.jl (%d lines)" % (len(rd("dwD5W.jl").splitlines()), len(rd("dwB5W.jl").splitlines()), len(rd("coupW5.jl").splitlines())))
