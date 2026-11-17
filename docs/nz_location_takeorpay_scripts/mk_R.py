import io
L = io.open("dwLeanN.jl", encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(L) if l.startswith("const VSM = ")); assert "mk(S, seed) = " in L[i] and L[i + 1].startswith("nd = mk(10, SEED)")
L[i] = L[i].replace("mk(S, seed) = ", "mk0(S, seed) = ")
add = '''# dwLeanR.jl (2026-11-17 night): the retailer may leave demand unserved.  Column u_j >= 0 in the demand row (sum_i (yR + yS) + u_j >= xi_j), retailer cost r u_j (r = value of a unit sold to the customer, env SC_R; 0 = off), leader gets nothing from u_j.
const RSELL = parse(Float64, E("SC_R", "0"))
function with_shortage(ndx, r)
    r > 0 || return ndx
    m, ny = nz_m(ndx), nz_ny(ndx); rd = ndx.meta[:rdem]; A = hcat(ndx.A, zeros(m, ncu)); for j in 1:ncu; A[rd(j), ny+j] = -1.0; end
    return NZData(ndx.name * "_short", A, vcat(ndx.c, fill(-r, ncu)), vcat(ndx.ell, zeros(ncu)), ndx.u, ndx.hrow, ndx.hcoef, ndx.xrow, ndx.g, ndx.W, ndx.wvec, ndx.c0, ndx.hU,
        ndx.nx, ndx.x_allowed, ndx.gamma, ndx.qx, ndx.S, ndx.q_hat, ndx.eps_hat, ndx.eps_tilde, ndx.beta, ndx.thetaU, ndx.piLU, ndx.piFU, ndx.lambdaU, ndx.varpiU, ndx.meta)
end
mk(S, seed) = with_shortage(mk0(S, seed), RSELL)'''
L.insert(i + 1, add); io.open("dwLeanR.jl", "w", encoding="utf-8", newline="").write("\n".join(L)); print("dwLeanR.jl written, mk at line", i + 1)
