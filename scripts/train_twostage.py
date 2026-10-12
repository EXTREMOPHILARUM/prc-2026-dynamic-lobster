"""Train and evaluate the two-stage model on the month-disjoint holdout.

    uv run python scripts/train_twostage.py
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score, root_mean_squared_error

from prc import config, data, features, train, twostage


def main() -> None:
    raw = features.departures_with_target(data.load_training()).reset_index(drop=True)
    X, y, cats = twostage.build_xy(raw)
    tr_idx, va_idx = train.month_disjoint_split(raw)
    print(f"rows {len(raw):,} | train {len(tr_idx):,} valid {len(va_idx):,}")

    models = twostage.fit(X.loc[tr_idx], y.loc[tr_idx], cats, X.loc[va_idx], y.loc[va_idx])

    pred = twostage.predict(models, X.loc[va_idx])
    yv = y.loc[va_idx].to_numpy()
    tail = yv > twostage.TAIL_THRESHOLD

    full = root_mean_squared_error(yv, pred)
    bulk = root_mean_squared_error(yv[~tail], pred[~tail])
    p_tail = models["clf"].predict(X.loc[va_idx][models["clf"].feature_name()])
    print(f"\nholdout RMSE  full {full:.2f}   bulk-only {bulk:.2f}")
    print(f"tail AUC {roc_auc_score(tail, p_tail):.4f}  |  predicted tail mass "
          f"{p_tail.sum():.0f} vs actual {tail.sum()}")

    va = raw.loc[va_idx].assign(y=yv, p=pred)
    per_airport = va.groupby("ADEP_mvt").apply(
        lambda d: pd.Series({
            "n": len(d),
            "rmse": root_mean_squared_error(d.y, d.p),
            "sq_share%": 100 * ((d.y - d.p) ** 2).sum() / ((va.y - va.p) ** 2).sum(),
        }),
        include_groups=False,
    ).sort_values("sq_share%", ascending=False)
    print("\n" + per_airport.round(1).to_string())

    month = pd.to_datetime(va[features.TIME_COL], utc=True).dt.month
    print("\nper-month RMSE:", {
        int(m): round(root_mean_squared_error(g.y, g.p), 2) for m, g in va.groupby(month)
    })

    twostage.save(
        models,
        {
            "rmse_full": float(full),
            "rmse_bulk": float(bulk),
            "features": list(X.columns),
            "categoricals": cats,
            "category_levels": features.category_levels(X, cats),
            "tail_threshold": twostage.TAIL_THRESHOLD,
        },
        config.MODELS / "twostage",
    )
    print(f"\nsaved {config.MODELS / 'twostage'}.*")


if __name__ == "__main__":
    main()
