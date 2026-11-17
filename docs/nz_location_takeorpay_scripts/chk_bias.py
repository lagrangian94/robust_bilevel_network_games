import csv
import numpy as np
OLD = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/expo/"
for cfg in ("expo15", "expo16"):
    for cv in ("0.5", "0.2"):
        o = []; pg = []
        for g in range(1, 16):
            v = []
            for s in range(1, 11):
                for r in csv.DictReader(open(OLD + "expoS_%s_g%d_s%d.csv" % (cfg, g, s))):
                    if r["cv"] == cv: v.append(100 * (float(r["JN"]) - float(r["VN"])) / abs(float(r["JN"])))
            o += v; pg.append(np.mean(v))
        o = np.array(o); pg = np.array(pg)
        print(cfg, "c/v", cv, "JN-VN %%: mean %+.2f (geo SE %.2f) median %+.2f  q10 %+.1f q90 %+.1f  share<0 %.0f%%  geos with mean<0: %d" % (o.mean(), pg.std(ddof=1) / np.sqrt(15), np.median(o), np.quantile(o, .1), np.quantile(o, .9), 100 * (o < 0).mean(), (pg < 0).sum()))
