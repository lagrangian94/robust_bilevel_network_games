# make the "retailer may leave demand unserved" copies of the Coupled / evaluation / decomposition / shipment scripts: same patch of mk() as dwLeanR.jl (mk_R.py)
import io
ADD = io.open("dwLeanR.jl", encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(ADD) if l.startswith("# dwLeanR.jl (2026-11-17 night)")); j = next(k for k, l in enumerate(ADD) if l.startswith("mk(S, seed) = with_shortage")); ADD = ADD[i:j + 1]
for src, dst in (("coupWN.jl", "coupWR.jl"), ("zsWN.jl", "zsWR.jl"), ("mechW.jl", "mechWR.jl"), ("flowW.jl", "flowWR.jl")):
    L = io.open(src, encoding="utf-8").read().split("\n")
    k = next(q for q, l in enumerate(L) if l.startswith("const VSM = ")); assert "mk(S, seed) = " in L[k] and L[k + 1].startswith("nd = mk(10, SEED)"), src
    L[k] = L[k].replace("mk(S, seed) = ", "mk0(S, seed) = "); L[k + 1:k + 1] = ADD
    s = "\n".join(L)
    if dst == "flowWR.jl":   # also write the unserved demand per customer (kind U): columns after the take-or-pay columns
        a = r'@printf("FLOW_DONE g%s seed %d: %.0f s\n", G, SS, time() - t0)'; assert s.count(a) == 1
        b = ('if RSELL > 0\n    open(SPD * "geo/$(OUT)/flow_g$(G)_s$(SS).csv", "a") do io\n        nyl = size(Y, 1)\n'
             '        for j in 1:ncul; println(io, join([SS, LAB[iN], 0, "U", 0, 0, 0, j, creg[j], 0, round(μ[j]; digits=1), 0, round(dot(qs, Y[nyl-ncul+j, :]); digits=3)], ";")); end\n    end\nend\n')
        s = s.replace(a, b + a)
    io.open(dst, "w", encoding="utf-8", newline="").write(s); print(dst, "ok")
