"""Leave-one-month-out calibration of the tail classifier.

The two-stage model predicted 0.452% tail mass on the 2026 ranking set against
0.285% actually observed in the 2025 holdout, inflating every prediction. That is an
out-of-time calibration failure, so measure it out of time: for each month, train the
classifier on the other eleven and compare predicted tail mass with actual.

The shrinkage factor is fitted on the ten months that are NOT the holdout (Jan, Jul),
so the holdout stays an honest test of it.

    uv run python scripts/calibrate_tail.py
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from prc import config, data, features, twostage

ROUNDS = 100  # close to the early-stopped optimum, avoids a nested validation split


def main() -> None:
    raw = features.departures_with_target(data.load_training()).reset_index(drop=True)
    X, y, cats = twostage.build_xy(raw)
    month = pd.to_datetime(raw[features.TIME_COL], utc=True).dt.month
    is_tail = (y > twostage.TAIL_THRESHOLD).astype(int)

    rows = []
    oof = np.full(len(raw), np.nan)
    for m in range(1, 13):
        trn, val = month != m, month == m
        clf = lgb.train(
            twostage.CLF_PARAMS,
            lgb.Dataset(X[trn], is_tail[trn], categorical_feature=cats),
            num_boost_round=ROUNDS,
        )
        p = clf.predict(X[val])
        oof[val.to_numpy()] = p
        rows.append({
            "month": m,
            "n": int(val.sum()),
            "pred_mass": float(p.sum()),
            "actual": int(is_tail[val].sum()),
            "ratio": float(p.sum() / max(is_tail[val].sum(), 1)),
        })
        print(f"  month {m:>2}: predicted {p.sum():7.0f}  actual {is_tail[val].sum():5d}  "
              f"ratio {rows[-1]['ratio']:.3f}")

    cal = pd.DataFrame(rows)
    fit_months = cal[~cal.month.isin([1, 7])]
    shrink = float(fit_months.actual.sum() / fit_months.pred_mass.sum())
    print(f"\nout-of-time ratio predicted/actual: mean {cal.ratio.mean():.3f}  "
          f"median {cal.ratio.median():.3f}")
    print(f"shrinkage fitted on the 10 non-holdout months: {shrink:.4f}")

    # honest test: apply it to the Jan+Jul holdout with the existing bulk/tail models
    models, _ = twostage.load(config.MODELS / "twostage")
    hold = month.isin([1, 7]).to_numpy()
    Xh, yh = X[hold], y[hold].to_numpy()
    bulk = models["bulk"].predict(Xh[models["bulk"].feature_name()])
    tail = np.maximum(models["tail"].predict(Xh[models["tail"].feature_name()]),
                      twostage.TAIL_THRESHOLD)
    p_raw = oof[hold]  # out-of-fold, so not fitted on these months

    def blended(s: float) -> np.ndarray:
        p = np.clip(p_raw * s, 0, 1)
        return np.clip(p * tail + (1 - p) * bulk, 0, None)

    print("\nholdout RMSE vs shrinkage applied to P(tail):")
    best = (1.0, np.inf)
    for s in [1.0, shrink, 0.9, 0.8, 0.7, 0.6, 0.5]:
        r = root_mean_squared_error(yh, blended(s))
        flag = "  <- fitted" if abs(s - shrink) < 1e-9 else ""
        print(f"  s={s:.4f}  RMSE {r:.2f}{flag}")
        if r < best[1]:
            best = (s, r)
    print(f"\nbest on holdout: s={best[0]:.4f} RMSE {best[1]:.2f}  "
          f"(fitted-blind choice s={shrink:.4f})")

    out = config.MODELS / "tail_calibration.json"
    out.write_text(json.dumps({
        "shrink": shrink,
        "per_month": rows,
        "holdout_rmse_at_shrink": float(root_mean_squared_error(yh, blended(shrink))),
    }, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
