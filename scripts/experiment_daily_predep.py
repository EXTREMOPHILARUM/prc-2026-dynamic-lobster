"""Daily reconciliation against EUROCONTROL's pre-departure delay (idea from team gracious-tornado's public repo).

EUROCONTROL `all_pre_departure_delays_<year>.csv` gives DLY_ALL_PRE_2 (minutes) per airport and take-off day. On 2025
labels it equals sum over the day's departures of max(0, off-block - schedule) = max(0, d - y) with d = MVT - SCHED
(corr 0.9996, median |gap| 2.8 min per airport-day). So K_g = 60 * DLY is a near-exact constraint on the day's targets.

Simple version: per airport-day, the additive shift s on ordinary rows (d <= DMAX) that makes the predicted sum match
K given the long rows' predicted contribution; rows then move by b * s, b fitted through the origin on the other
months (leave-one-month-out), s clipped to +/-SCLIP. Paired against the v48-equivalent OOF.

    uv run python -u scripts/experiment_daily_predep.py PDD_2025_CSV
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from prc import config, evaluation, matrix

DMAX = 3600.0
SCLIP = 900.0


def day_shift(d, p, ordinary, K):
    fixed = np.maximum(0, d[~ordinary] - p[~ordinary]).sum()
    do, po = d[ordinary], p[ordinary]
    f = lambda s: fixed + np.maximum(0, do - np.maximum(po + s, 0)).sum() - K
    lo, hi = -SCLIP, SCLIP
    if f(lo) < 0:
        return lo
    if f(hi) > 0:
        return hi
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


BIN = 60.0


def tlik(x, scale, df=3.0):
    return (1 + (x / scale) ** 2 / df) ** (-(df + 1) / 2)


def long_posterior(d, p, t0, R, sigma):
    """P(recorded at schedule) per long row given the long rows' total contribution R (seconds).

    Each long row i either copies the schedule (y = d, contributes 0) or is a real taxi (y = t0, contributes d - t0);
    prior pi_i = (p_i - t0) / (d_i - t0) keeps the prior mean at p_i. Exact over the 2^n combinations by convolution
    on a BIN grid, R observed with Student-t(3) noise of scale sigma."""
    n = len(d)
    pi = np.clip((p - t0) / (d - t0), 0.01, 0.99)
    c = np.maximum(0, np.round((d - t0) / BIN)).astype(int)
    L = int(c.sum()) + 1
    grid = np.arange(L) * BIN
    h = tlik(R - grid, sigma)
    post = np.empty(n)
    for i in range(n):
        dist = np.zeros(L); dist[0] = 1.0
        for j in range(n):
            if j == i:
                continue
            nxt = dist * pi[j]
            if c[j]:
                nxt[c[j]:] += dist[:-c[j]] * (1 - pi[j])
            else:
                nxt += dist * (1 - pi[j])
            dist = nxt
        l_copy = (dist * h).sum()
        l_real = (dist[: L - c[i]] * h[c[i]:]).sum() if c[i] else l_copy
        post[i] = pi[i] * l_copy / max(pi[i] * l_copy + (1 - pi[i]) * l_real, 1e-300)
    return post


def main():
    raw, _, _ = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    y = raw[config.TARGET].to_numpy("float64"); month = evaluation.movement_month(raw).to_numpy()
    p = raw[[config.ID_COL]].merge(pd.read_parquet(config.DATA_INTERIM / "v48_equiv_oof.parquet"), on=config.ID_COL,
                                   how="left")["prediction"].to_numpy("float64")
    assert np.isfinite(p).all()
    mvt = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True)
    d = (mvt - pd.to_datetime(raw["SCHED_TIME_UTC_mvt"], utc=True)).dt.total_seconds().to_numpy()
    day = mvt.dt.tz_localize(None).dt.normalize()
    pdd = pd.read_csv(sys.argv[1], parse_dates=["FLT_DATE"])
    key = pd.DataFrame({"a": raw["ADEP_mvt"].astype(str).to_numpy(), "day": day.to_numpy()})
    K = key.merge(pdd.rename(columns={"APT_ICAO": "a", "FLT_DATE": "day"})[["a", "day", "DLY_ALL_PRE_2"]],
                  on=["a", "day"], how="left")["DLY_ALL_PRE_2"].to_numpy() * 60
    ok = np.isfinite(d) & np.isfinite(K)
    ordinary = ok & (d <= DMAX)
    s = np.zeros(len(y))
    grp = key.groupby(["a", "day"]).indices
    for idx in grp.values():
        idx = idx[ok[idx]]
        if len(idx):
            s[idx] = day_shift(d[idx], p[idx], ordinary[idx], K[idx][0])
    delayed = ordinary & (d - p > 0)
    r = y - p
    rmse = lambda e, m: float(np.sqrt(np.mean(e[m] ** 2)))
    for label, cls in (("delayed", delayed), ("ordinary other", ordinary & ~delayed)):
        print(label, int(cls.sum()), "corr(r, s)", round(np.corrcoef(r[cls & (np.abs(r) < 3600)], s[cls & (np.abs(r) < 3600)])[0, 1], 3))
    cand = p.copy()
    for m in range(1, 13):
        tr, va = month != m, month == m
        for cls in (delayed, ordinary & ~delayed):
            fit = tr & cls & (np.abs(r) < 3600) & (s != 0)
            b = float(np.clip((r[fit] * s[fit]).sum() / (s[fit] ** 2).sum(), -0.5, 2.5))
            cand[va & cls] = np.maximum(p[va & cls] + b * s[va & cls], 0)
            if m == 1:
                print("b (fold 1)", round(b, 3))
    # long rows: copy-vs-held posterior from the day's remaining sum
    import os
    LMIN = float(os.environ.get("LMIN", DMAX))
    long_ = ok & (d > LMIN) & (raw["ADEP_mvt"].astype(str).to_numpy() == "LIRF")
    lng = raw.assign(_y=y, _d=d)[long_ & (np.abs(y - d) > 120) & (y < 7200)].groupby("ADEP_mvt", observed=True)["_y"].median()
    t0 = raw["ADEP_mvt"].astype(str).map(lng.to_dict()).fillna(900.0).to_numpy()
    sig = {}
    for (a, _), idx in grp.items():
        idx = idx[ok[idx]]
        if len(idx):
            o = idx[~long_[idx]]; l = idx[long_[idx]]
            e = K[idx][0] - np.maximum(0, d[l] - y[l]).sum() - np.maximum(0, d[o] - cand[o]).sum()
            sig.setdefault(a, []).append(e)
    sig = {a: 1.4826 * np.median(np.abs(np.array(v) - np.median(v))) for a, v in sig.items()}
    print("sigma by airport (s)", {a: round(v) for a, v in sig.items()})
    cand2 = cand.copy(); nlong = 0
    for (a, _), idx in grp.items():
        idx = idx[ok[idx]]
        l = idx[long_[idx]]
        if a != "LIRF" or not len(l) or len(l) > 60:
            continue
        o = idx[~long_[idx]]
        R = K[idx][0] - np.maximum(0, d[o] - cand[o]).sum()
        real = np.minimum(cand[l], t0[l])
        pc = long_posterior(d[l], cand[l], real, R, sig[a])
        cand2[l] = pc * d[l] + (1 - pc) * real; nlong += len(l)
    print("long rows moved", nlong)
    for label, msk in (("twelve", np.ones(len(y), bool)), ("Jan+Jul", np.isin(month, (1, 7))), ("Feb+Jun", np.isin(month, (2, 6))),
                       ("non-scored", ~np.isin(month, (1, 7)))):
        print(f"LONG {label:10s} {rmse(y - cand, msk):8.3f} -> {rmse(y - cand2, msk):8.3f} ({rmse(y - cand2, msk) - rmse(y - cand, msk):+.3f})")
    print("LONG by airport dSSE (M)", {a: round(float(((y - cand2) ** 2 - (y - cand) ** 2)[raw["ADEP_mvt"].astype(str).to_numpy() == a].sum()) / 1e6, 1) for a in sig})
    dl = {m: float(((y - cand2) ** 2 - (y - cand) ** 2)[month == m].sum()) / 1e6 for m in range(1, 13)}
    print("LONG dSSE by month (M)", {m: round(v, 1) for m, v in dl.items()}, "months won", sum(v < 0 for v in dl.values()))
    dd2 = np.sort((y - cand2) ** 2 - (y - cand) ** 2)
    print("LONG drop-1/10/50 dSSE (M)", [round(dd2[k:].sum() / 1e6, 1) for k in (1, 10, 50)])
    dsse = {m: float(((y - cand) ** 2 - (y - p) ** 2)[month == m].sum()) / 1e6 for m in range(1, 13)}
    for label, msk in (("twelve", np.ones(len(y), bool)), ("Jan+Jul", np.isin(month, (1, 7))), ("Feb+Jun", np.isin(month, (2, 6))),
                       ("non-scored", ~np.isin(month, (1, 7)))):
        print(f"{label:10s} {rmse(y - p, msk):8.3f} -> {rmse(y - cand, msk):8.3f} ({rmse(y - cand, msk) - rmse(y - p, msk):+.3f})")
    print("dSSE by month (M)", {m: round(v, 1) for m, v in dsse.items()}, "months won", sum(v < 0 for v in dsse.values()))
    dd = np.sort((y - cand) ** 2 - (y - p) ** 2)
    print("drop-1/10/50 dSSE (M)", [round(dd[k:].sum() / 1e6, 1) for k in (1, 10, 50)])


if __name__ == "__main__":
    main()
