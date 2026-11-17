import pickle, numpy as np
from possession import config, model, refine
ids = model.external_matches(config.load_params())
D = {m: pickle.load(open(f"outputs/logs/refine_cache/{m}_seqT_to.pkl", "rb")) for m in ids}
for W in (38, 100):
    C = {}
    for m in ids:
        d = D[m]; F = refine.frame_table(d["bt"], d["e"])
        X, bi, cf, bs = refine.candidates(d["lab"], d["halves"], F, W)
        C[m] = (X, bi, cf, bs, refine.targets(np.asarray(d["gt"]), bs, bi, cf))
    for kind in ("turnover", "restart", "stoppage"):
        tot = inwin = here = top = obs_here = 0; dist = []
        for m in ids:
            X, bi, cf, bs, y = C[m]
            mdl = refine.fit([(C[o][0], C[o][4]) for o in ids if o != m]) if W == 38 else None
            p = mdl.predict_proba(X)[:, 1] if mdl else None
            for i, (t, k, team, lo, hi) in enumerate(bs):
                if k != kind: continue
                idx = np.flatnonzero(bi == i); tot += 1
                if not y[idx].any(): continue
                inwin += 1
                h = idx[cf[idx] == t]; here += bool(len(h) and y[h[0]])
                if p is not None: top += bool(y[idx[np.argmax(p[idx])]])
        print(f"W{W} {kind:9} boundaries {tot:5d}  true one within window {inwin/tot:.2f}  of those: already right (+-2f) {here/max(inwin,1):.2f}" + (f"  refiner's pick right {top/max(inwin,1):.2f}" if W == 38 else ""))
