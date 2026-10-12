"""Fit day-offset band rates on all 12 months and save a deployable wrapper."""
from __future__ import annotations

import json

from prc import config, data, dayoffset, features


def main() -> None:
    raw = features.departures_with_target(data.load_training()).reset_index(drop=True)
    X, _ = features.build(raw)
    y = raw[config.TARGET].astype("float64").to_numpy()
    label = dayoffset.day_offset_label(y)
    bands = dayoffset.band(raw, X)
    rates = dayoffset.fit_rates(bands, label)
    print(f"positives {label.sum()} | rates {rates}")

    meta = json.loads((config.MODELS / "mechanism.json").read_text())
    meta |= {
        "kind": "mechanism_day",
        "ordinary_stem": "mechanism",
        "day_rates": rates,
        "day_threshold": dayoffset.DAY_THRESHOLD,
        "prior_strength": dayoffset.PRIOR_STRENGTH,
        "gap_edges_h": list(dayoffset.GAP_EDGES_H),
    }
    (config.MODELS / "mechanism_day.json").write_text(json.dumps(meta, indent=2))
    print("saved models/mechanism_day.json")


if __name__ == "__main__":
    main()
