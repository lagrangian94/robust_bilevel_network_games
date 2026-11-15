import csv, glob, re, collections, sys
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/tk/"
SC = [(0.0, 2.5, "0.5"), (150.0, 2.0, "0.4"), (450.0, 1.0, "0.2")]
for cfg in sys.argv[1:]:
    fs = sorted(glob.glob(O + "vn_%s_g*_s*.csv" % cfg))
    for q0, c, lab in SC:
        cnt = collections.Counter(); nst = collections.Counter(); gap = []
        for f in fs:
            R = []
            for r in csv.DictReader(open(f, encoding="utf-8")):
                pr = [p.split(":") for p in r["x"].strip("{}").split(";")]
                lv = [float(b) for _, b in pr]; R.append((float(r["V0"]) + q0 * len(lv) + c * sum(lv), tuple(int(a) for a, _ in pr), tuple(lv)))
            R.sort(); v, s, lv = R[0]
            for b in lv: cnt[int(b)] += 1
            nst[len(lv)] += 1
            # best decision with a different total capacity: how much worse (percent of |v|)
            alt = min(r[0] for r in R if sum(r[2]) != sum(lv)); gap.append(100 * (alt - v) / abs(v))
        tot = sum(cnt.values())
        print("%s c/v %s: runs %d | tiers at N: %s | stores %s | next-best with other total capacity: +%.2f%% (median), min +%.2f%%" % (
            cfg, lab, len(fs), ", ".join("%d: %.0f%%" % (k, 100 * cnt[k] / tot) for k in sorted(cnt)), dict(nst), sorted(gap)[len(gap) // 2], min(gap)))
