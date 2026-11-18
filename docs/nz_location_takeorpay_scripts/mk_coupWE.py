import io
s = io.open("coupW.jl", encoding="utf-8").read()
a = 'const NPT = parse(Int, E("CP_NPT", "30"))'; assert s.count(a) == 1; s = s.replace(a, 'const CLAB = ENV["CP_LAB"]   # coupWE.jl (2026-11-18): model label and output file names from CP_LAB, the files of the 0.1 run are not touched\n' + a)
a = 'open(SPD * "geo/dw/cp_$(tag0).csv", "w") do io'; assert s.count(a) == 1; s = s.replace(a, 'open(SPD * "geo/dw/cpx_$(CCFG)_$(CLAB)_g$(GG)_s$(SEED).csv", "w") do io')
i = s.index('sel = readlines(SPD * "geo/dw/sel_$(tag0).csv")'); j = s.index('@printf("CP_DONE')
new = '''open(SPD * "geo/dw/selx_$(CCFG)_$(CLAB)_g$(GG)_s$(SEED).csv", "w") do io
    println(io, "scen,q0,c,model,x")
    for (k, (q0, c)) in enumerate(SCEN)
        ci = sortperm(collect(zip(VC .+ CST[k][cand], LAB[cand])))[1]; println(io, k, ",", q0, ",", c, ",", CLAB, ",", LAB[cand[ci]])
        @printf("SELC scen %d: %s=%s cap %.0f (V0 %.2f, VC %.2f)\n", k, CLAB, LAB[cand[ci]], CAP[cand[ci]], V0[ci], VC[ci])
    end
end
'''
s = s[:i] + new + s[j:]; io.open("coupWE.jl", "w", encoding="utf-8", newline="").write(s); print("coupWE.jl written")
