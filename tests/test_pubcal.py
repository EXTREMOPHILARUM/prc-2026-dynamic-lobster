import numpy as np
import pandas as pd

from prc import pubcal

PUBLISHED = pd.DataFrame({
    "apt": ["AAAA", "AAAA", "BBBB"],
    "year": [2026, 2026, 2026],
    "month": [1, 7, 1],
    "pub": [1100.0, 900.0, 800.0],
    "valid": [0.99, 0.95, 1.0],
})


def test_apply_shifts_valid_months_to_published_minus_offset():
    airport = np.array(["AAAA", "AAAA", "AAAA", "AAAA", "BBBB", "CCCC"])
    t = pd.to_datetime(["2026-01-03", "2026-01-04", "2026-01-05", "2026-07-01", "2026-01-02", "2026-01-02"], utc=True)
    pred = np.array([900.0, 1100.0, 5000.0, 700.0, 850.0, 600.0])
    out, rows = pubcal.apply(pred, airport, t, PUBLISHED, {"AAAA": 50.0, "BBBB": 0.0})
    # AAAA Jan: rows under the cap average 1000, target 1100 - 50 -> +50; the 5000s row stays
    # AAAA Jul: valid 0.95 < VMIN, untouched; BBBB Jan: target 800 -> -50; CCCC: no published row
    assert out.tolist() == [950.0, 1150.0, 5000.0, 700.0, 800.0, 600.0]
    assert rows.tolist() == [True, True, False, False, True, False]
    assert pred.tolist() == [900.0, 1100.0, 5000.0, 700.0, 850.0, 600.0]


def test_apply_skips_airports_without_an_offset():
    t = pd.to_datetime(["2026-01-02"], utc=True)
    out, rows = pubcal.apply(np.array([850.0]), np.array(["BBBB"]), t, PUBLISHED, {})
    assert out.tolist() == [850.0] and not rows.any()


def test_fit_offsets_uses_valid_months_and_rows_under_cap():
    airport = np.array(["AAAA", "AAAA", "AAAA", "AAAA"])
    t = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-07-01"], utc=True)
    y = np.array([1000.0, 1200.0, 90000.0, 500.0])
    pred = np.array([1000.0, 1000.0, 5000.0, 500.0])
    # Jan: true mean of rows predicted under the cap is 1100, published 1100 -> 0; Jul is below VMIN
    assert pubcal.fit_offsets(airport, t, y, pred, PUBLISHED) == {"AAAA": 0.0}
