import numpy as np
import pandas as pd

from prc import lirfbands


def test_garbled_flight_matches_doubled_prefixes_only():
    s = pd.Series(["ITYTY672", "WMTT5232", "RYRR90BN", "QFAQFA6", "CHHHH438",
                   "DLH8LA", "RYR14WC", "EJU42AY", "ITY332", None, "AB12"])
    assert lirfbands.garbled_flight(s).tolist() == [True] * 5 + [False] * 6


def test_garbled_copy_pins_only_in_scope_rows():
    h = 3600.0
    X = pd.DataFrame({
        "ADEP_mvt": ["LIRF", "LIRF", "LIRF", "LIRF", "LIRF", "LEMD"],
        lirfbands.MISSING_COL: [1, 1, 0, 1, 1, 1],
        lirfbands.SCHED_COL: [5 * h, 5 * h, 5 * h, 4 * h, 9 * h, 5 * h],
    })
    raw = pd.DataFrame({"FLIGHT_mvt": ["ITYTY672", "ITY672", "ITYTY672", "ITYTY672", "ITYTY672", "ITYTY672"]})
    pred = np.full(6, 1000.0)
    out = lirfbands.garbled_copy(pred, raw, X)
    # only row 0: LIRF, no record, garbled, gap in (4h, 8h]; 4h itself is outside
    assert out.tolist() == [5 * h, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    assert pred.tolist() == [1000.0] * 6
