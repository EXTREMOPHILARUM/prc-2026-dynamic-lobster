"""Audit NM-clock donors in supplied movements, without reading taxi labels.

Exact flight IDs and flight/route/scheduled-date keys are diagnostics, not an
imputation policy. Known-clock controls use arrival-only donors to exclude self
matches. Multiple distinct donor clocks make a key ambiguous and unusable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prc import config, data

COLUMNS = [config.ID_COL, "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt",
           "PHASE_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]


def donor_lookup(donors, queries, keys):
    donors = donors.dropna(subset=[*keys, "AOBT_3_flt"])
    grouped = donors.groupby(keys, observed=True)["AOBT_3_flt"].agg(["nunique", "first"])
    result = queries.merge(grouped, on=keys, how="left", validate="many_to_one")
    return result


def audit(frame):
    frame = frame.copy()
    for col in ("SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    frame["scheduled_day"] = frame["SCHED_TIME_UTC_mvt"].dt.floor("D")
    dep = frame[frame.PHASE_mvt.eq("DEP")]
    absent = dep[dep.AOBT_3_flt.isna()]
    known = dep[dep.AOBT_3_flt.notna()]
    report = {"departures": len(dep), "missing_nm": len(absent),
              "missing_nm_with_flight_id": int(absent.FLIGHT_ID_mvt.notna().sum()),
              "missing_nm_with_flight_name": int(absent.FLIGHT_mvt.notna().sum())}
    for name, keys in (
        ("flight_id", ["FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt"]),
        ("flight_route_scheduled_day", ["FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "scheduled_day"]),
    ):
        joined = donor_lookup(frame, absent, keys)
        unique = joined["nunique"].eq(1)
        recovered = joined.loc[unique]
        taxi_proxy = (recovered.MVT_TIME_UTC_mvt - recovered["first"]).dt.total_seconds()
        control = donor_lookup(frame[frame.PHASE_mvt.eq("ARR")], known, keys)
        control = control[control["nunique"].eq(1)]
        error = (control.AOBT_3_flt - control["first"]).dt.total_seconds().abs()
        report[name] = {
            "missing_unique_donor": len(recovered),
            "missing_ambiguous_donor": int(joined["nunique"].gt(1).sum()),
            "missing_unique_by_airport": recovered.ADEP_mvt.value_counts().to_dict(),
            "recovered_proxy_quantiles_s": {
                str(q): float(taxi_proxy.quantile(q)) if len(taxi_proxy) else None
                for q in (0, 0.5, 0.95, 1)
            },
            "known_clock_arrival_controls": len(control),
            "control_exact_fraction": float(error.eq(0).mean()) if len(error) else None,
            "control_over_60s": int(error.gt(60).sum()),
            "control_abs_error_p99_s": float(np.quantile(error, 0.99)) if len(error) else None,
        }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {}
    for kind, loader in (("training", data.load_training), ("ranking", data.load_ranking)):
        result[kind] = audit(loader(columns=COLUMNS))
        print(kind, json.dumps(result[kind], indent=2), flush=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
