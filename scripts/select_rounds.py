"""Select component round counts WITHOUT looking at the ranking months.

DEFAULT_ROUNDS were early-stopped on the January+July holdout, and the 12-fold OOF then
reused them for every fold - including folds 1 and 7. That makes the OOF estimate for the
two months we are actually scored on optimistic, because their iteration counts were chosen
by looking at their own labels.

This selects rounds on an inner holdout drawn from months disjoint from January and July,
so folds 1 and 7 become honest. The inner-holdout months themselves are then mildly
optimistic, which is the right place to put the bias: we are not scored on April or October.

    uv run python scripts/select_rounds.py                 # inner holdout = months 4, 10
    uv run python scripts/select_rounds.py --inner 3 9
"""
from __future__ import annotations

import argparse
import json

from prc import config, data, decomposition, evaluation, features

RANKING_MONTHS = (1, 7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner", type=int, nargs="+", default=[4, 10])
    args = parser.parse_args()

    overlap = set(args.inner) & set(RANKING_MONTHS)
    if overlap:
        raise SystemExit(
            f"inner holdout months {sorted(overlap)} overlap the ranking months "
            f"{RANKING_MONTHS} - that reintroduces the leak this script exists to remove"
        )

    raw = features.departures_with_target(data.load_training()).reset_index(drop=True)
    X, cats = features.build(raw)
    X = X.drop(columns=["month"])
    y = raw[config.TARGET].astype("float32")
    month = evaluation.movement_month(raw)

    inner = month.isin(args.inner)
    print(f"inner holdout months {args.inner}: {inner.sum():,} rows | "
          f"train {(~inner).sum():,} rows")

    models = decomposition.fit(
        X.loc[~inner], y.loc[~inner], cats, X.loc[inner], y.loc[inner]
    )
    rounds = {name: int(model.best_iteration) for name, model in models.items()}
    print(f"\nselected rounds: {rounds}")
    print(f"previous (leaky) rounds: {decomposition.DEFAULT_ROUNDS}")

    out = config.DATA_INTERIM / "component_rounds.json"
    out.write_text(json.dumps({"inner_months": args.inner, "rounds": rounds}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
