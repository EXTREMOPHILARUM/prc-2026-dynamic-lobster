"""Guard against self-matching and optimistic concentration diagnostics."""
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
audit = run_path(str(SCRIPTS / "audit_nm_linkage.py"))["audit"]
delta_report = run_path(str(SCRIPTS / "experiment_architecture.py"))["delta_report"]


def test_linkage_rejects_conflicting_clocks_and_excludes_self_controls():
    frame = pd.DataFrame({
        "MVT_ID_mvt": [1, 2, 3, 4, 5],
        "FLIGHT_ID_mvt": [None, 20, 30, None, 50],
        "FLIGHT_mvt": ["A", "A", "A", "B", "B"],
        "ADEP_mvt": ["LIRF"] * 5,
        "ADES_mvt": ["LFPG"] * 5,
        "PHASE_mvt": ["DEP", "ARR", "ARR", "DEP", "ARR"],
        "SCHED_TIME_UTC_mvt": ["2025-02-01T10:00:00Z"] * 5,
        "MVT_TIME_UTC_mvt": ["2025-02-01T11:00:00Z"] * 5,
        "AOBT_3_flt": [None, "2025-02-01T10:10:00Z", "2025-02-01T10:20:00Z",
                       None, "2025-02-01T10:30:00Z"],
    })
    result = audit(frame)
    match = result["flight_route_scheduled_day"]
    assert match["missing_unique_donor"] == 1
    assert match["missing_ambiguous_donor"] == 1
    assert match["recovered_proxy_quantiles_s"]["0.5"] == 1800
    # A known departure cannot validate itself, even when it has a unique ID.
    frame.loc[0, "AOBT_3_flt"] = "2025-02-01T10:10:00Z"
    frame.loc[0, "FLIGHT_ID_mvt"] = 99
    result = audit(frame)
    assert result["flight_id"]["known_clock_arrival_controls"] == 0


def test_concentration_check_exposes_one_row_gain_and_empty_segments():
    n = 100
    frame = pd.DataFrame({
        "TAXITIME_SEC_mvt": np.zeros(n),
        "control": [100.0, *np.zeros(n - 1)],
        "candidate": [0.0, *np.ones(n - 1)],
        "aobt_missing": False,
        "airport": "LIRF",
        "airport_day": ["LIRF:2025-02-01", *["LIRF:2025-02-02"] * (n - 1)],
        "month": 2,
    })
    report = delta_report(frame, "candidate")
    assert report["all"]["delta"] < 0
    assert report["drop_1"]["delta"] > 0
    assert report["drop_50"]["delta"] > 0
    assert report["drop_best_airport_day"]["delta"] > 0
    assert report["clock_missing"]["n"] == 0
    assert report["clock_missing"]["delta"] is None
