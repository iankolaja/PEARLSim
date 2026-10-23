import numpy as np
import pytest

from pearlsim.rod_worth import INSERTED, WITHDRAWN, WorthCurve, critical_position, isotonic

DEPTHS = np.array([369.47, 320.0, 280.0, 240.0, 215.0, 190.0, 150.0, 110.0, 60.0])


def s_curve(z, total=2000.0):
    """Textbook S-curve: w = W (x - sin(2 pi x) / (2 pi)), x = insertion fraction."""
    x = (WITHDRAWN - np.asarray(z)) / (WITHDRAWN - INSERTED)
    return total * (x - np.sin(2 * np.pi * x) / (2 * np.pi))


def test_isotonic_pools_violations_and_keeps_monotone_data():
    assert isotonic([0, 2, 1, 3]).tolist() == [0, 1.5, 1.5, 3]
    assert isotonic([0, 1, 2]).tolist() == [0, 1, 2]


def test_curve_reproduces_a_smooth_s_curve_between_points():
    curve = WorthCurve(DEPTHS, s_curve(DEPTHS))
    z = np.linspace(60, 369.47, 200)
    assert np.max(np.abs(curve(z) - s_curve(z))) < 0.01 * curve.total
    assert curve.total == pytest.approx(2000.0)
    assert np.all(np.diff(curve(z[::-1])) >= -1e-9)  # non-decreasing with insertion


def test_noisy_points_become_monotone_and_invertible():
    rng = np.random.default_rng(0)
    w = s_curve(DEPTHS) + rng.normal(0, 60, DEPTHS.size) * (DEPTHS < WITHDRAWN)
    curve = WorthCurve(DEPTHS[::-1], w[::-1], sigma=np.full(DEPTHS.size, 60.0))  # any input order
    assert np.all(np.diff(curve.w) >= 0) and curve.w[0] == 0
    for worth in (0.0, 300.0, 1000.0, curve.total):
        assert float(curve(curve.position_for_worth(worth))) == pytest.approx(worth, abs=1e-6 * curve.total + 1e-6)


def test_critical_position_statuses_and_margin():
    curve = WorthCurve(DEPTHS, s_curve(DEPTHS))
    z, status, margin = critical_position(curve, rho_step=500.0, z_act=WITHDRAWN)
    assert status == "critical" and float(curve(z)) == pytest.approx(500.0, abs=1e-3)
    assert margin == pytest.approx(1500.0)
    # already inserted to 215 cm: the step's excess adds to the worth of that insertion
    z2, _, _ = critical_position(curve, rho_step=200.0, z_act=215.0)
    assert float(curve(z2)) == pytest.approx(200.0 + float(curve(215.0)), abs=1e-3) and z2 < 215.0
    assert critical_position(curve, -50.0, WITHDRAWN)[1] == "subcritical_rods_out"
    _, status, margin = critical_position(curve, 2500.0, WITHDRAWN)
    assert status == "supercritical_rods_in" and margin == pytest.approx(-500.0)


def test_curve_needs_the_withdrawn_point():
    with pytest.raises(ValueError, match="withdrawn"):
        WorthCurve(DEPTHS[1:], s_curve(DEPTHS[1:]))
