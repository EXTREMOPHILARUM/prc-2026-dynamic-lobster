"""Fit the LIRF band table on all twelve months and write the v15 wrapper model.

    uv run python scripts/fit_lirfbands.py --inner mechanism_noid_arr --out mechanism_v15
"""
from __future__ import annotations

import argparse
import json

from prc import config, lirfbands, matrix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", default="mechanism_noid_arr")
    ap.add_argument("--out", default="mechanism_v15")
    ap.add_argument("--expert", default=None, help="stem of the NM-free expert for LIRF rows under the band scope")
    args = ap.parse_args()
    raw, X, _ = matrix.build("train", use_weather=True, use_sequencing=True)
    table = lirfbands.fit(X, raw[config.TARGET].to_numpy(dtype="float64"))
    inner = json.loads((config.MODELS / f"{args.inner}.json").read_text())
    meta = {
        "kind": "mechanism_bands",
        "inner_stem": args.inner,
        "band_table": table,
        "lobt_clip": True,
        "lobt_bound": lirfbands.LOBT_BOUND,
        "short_expert_stem": args.expert,
        "use_weather": inner.get("use_weather"),
        "use_sequencing": inner.get("use_sequencing"),
        "use_arrivals": inner.get("use_arrivals"),
        "external_data": inner.get("external_data"),
    }
    (config.MODELS / f"{args.out}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(table, indent=2))
    print(f"saved models/{args.out}.json wrapping {args.inner}")


if __name__ == "__main__":
    main()
