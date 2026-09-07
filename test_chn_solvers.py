"""Correctness tests for :mod:`chn_solvers`.

Run with ``python -m pytest test_chn_solvers.py -q`` or simply
``python test_chn_solvers.py``.
"""

from __future__ import annotations

import math

import numpy as np

import chn_solvers as chn


RNG = np.random.default_rng(20260828)


def poly_from_roots(roots) -> list[complex]:
    """Monic polynomial with the given roots, descending coefficients."""
    p = np.array([1.0 + 0.0j])
    for r in roots:
        p = np.convolve(p, [1.0 + 0.0j, -complex(r)])
    return list(p)


def match_error(found, true) -> float:
    """Largest distance from a true root to its nearest approximation."""
    if not found:
        return float("inf")
    return max(min(abs(complex(t) - complex(f)) for f in found) for t in true)


# --------------------------------------------------------------------------- #
def test_taylor_coefficients():
    """Repeated synthetic division must reproduce p^(j)(z)/j!."""
    coeffs = [2.0, -3.0, 0.5, 7.0, -1.0]
    z = 0.37 - 0.82j
    got = chn.taylor_coefficients(coeffs, z)
    # Reference: expand p(z + h) symbolically with numpy.
    n = len(coeffs) - 1
    want = []
    for j in range(n + 1):
        d = coeffs
        for _ in range(j):
            d = np.polyder(d)
        want.append(np.polyval(d, z) / math.factorial(j))
    for a, b in zip(got, want):
        assert abs(a - b) < 1e-12, (a, b)


def test_kalantari_bound_contains_roots():
    """Every root must lie inside U_2."""
    for _ in range(200):
        deg = int(RNG.integers(3, 20))
        c = list(RNG.normal(size=deg + 1))
        if abs(c[0]) < 1e-3:
            c[0] = 1.0
        r = chn.kalantari_bound(c)
        assert max(abs(z) for z in np.roots(c)) <= r + 1e-9


def test_rnm_strictly_decreases_modulus():
    """The closed RNM form must reduce |p| at every non-critical point."""
    coeffs = poly_from_roots(RNG.normal(size=7) + 1j * RNG.normal(size=7))
    failures = 0
    trials = 0
    for _ in range(2000):
        z = complex(RNG.uniform(-3, 3), RNG.uniform(-3, 3))
        p, dp = chn.horner_with_derivative(coeffs, z)
        if abs(p) < 1e-12 or abs(dp) < 1e-8:
            continue
        trials += 1
        step = chn.rnm_step(coeffs, z)
        assert step is not None
        if abs(chn.horner(coeffs, z + step)) >= abs(p):
            failures += 1
    assert trials > 500
    assert failures == 0, f"{failures}/{trials} RNM steps failed to descend"


def test_chn_monotone_modulus():
    """|p(z_t)| must be strictly decreasing along a CHN orbit."""
    coeffs = poly_from_roots([0.3, -0.7, 1.2 + 0.4j, 1.2 - 0.4j, -1.5])
    z = 2.0 + 1.7j
    previous = abs(chn.horner(coeffs, z))
    for _ in range(60):
        res = chn.solve_chn(coeffs, z, max_iter=1)
        z = res.roots[0]
        current = abs(chn.horner(coeffs, z))
        assert current <= previous + 1e-300
        previous = current


def test_smale_cycle_is_broken():
    """Newton cycles on z^3 - 2z + 2 from z0 = 0; CHN must not."""
    coeffs = [1.0, 0.0, -2.0, 2.0]
    res = chn.solve_chn(coeffs, 0.0 + 0.0j, max_iter=400)
    assert res.converged
    assert abs(chn.horner(coeffs, res.roots[0])) < 1e-10


def test_all_roots_competitive_with_lapack():
    """Accuracy must track LAPACK on the same instance.

    An absolute forward-error threshold would be the wrong test at high
    degree: random real-rooted polynomials of degree 20+ are genuinely
    ill-conditioned, and no backward-stable method can do better than the
    conditioning allows.  The meaningful requirement is that CHN stays within
    a small factor of the companion-matrix eigensolver on each instance.
    """
    for deg in (5, 10, 15, 20, 25):
        for _ in range(15):
            true = RNG.uniform(-1, 1, deg)
            coeffs = poly_from_roots(true)
            reference = match_error([complex(r) for r in np.roots(coeffs)], true)
            for truncated in (False, True):
                res = chn.solve_all_chn(coeffs, truncated=truncated)
                err = match_error(res.roots, true)
                budget = max(1e-8, 25.0 * reference)
                assert err <= budget, (
                    f"degree {deg}, truncated={truncated}: "
                    f"error {err:.2e} vs LAPACK {reference:.2e}"
                )


def test_backward_error_is_machine_level():
    """Every reported root must be a root of a nearby polynomial."""
    for deg in (8, 16, 24):
        for _ in range(10):
            true = RNG.uniform(-1, 1, deg)
            coeffs = poly_from_roots(true)
            res = chn.solve_all_chn(coeffs)
            for r in res.roots:
                residual = abs(chn.horner(coeffs, r))
                assert residual <= 1e4 * chn.backward_error_bound(coeffs, r)


def test_truncated_matches_full():
    """CHN-T must find the same roots as CHN."""
    for _ in range(20):
        deg = int(RNG.integers(4, 16))
        true = RNG.uniform(-1, 1, deg)
        coeffs = poly_from_roots(true)
        full = chn.solve_all_chn(coeffs, truncated=False)
        trunc = chn.solve_all_chn(coeffs, truncated=True)
        assert match_error(full.roots, true) < 1e-6
        assert match_error(trunc.roots, true) < 1e-6


def test_complex_roots_recovered():
    """Conjugate pairs that Yuksel discards must be found."""
    true = [0.4 + 0.9j, 0.4 - 0.9j, -1.1 + 0.2j, -1.1 - 0.2j, 0.25]
    coeffs = poly_from_roots(true)
    res = chn.solve_all_chn(coeffs)
    assert match_error(res.roots, true) < 1e-8


def test_yuksel_finds_real_roots():
    """Baseline sanity: Yuksel recovers real roots on its own turf."""
    for _ in range(20):
        deg = int(RNG.integers(3, 15))
        true = RNG.uniform(-1, 1, deg)
        coeffs = [c.real for c in poly_from_roots(true)]
        res = chn.solve_yuksel(coeffs, -1.5, 1.5)
        assert match_error(res.roots, true) < 1e-6


def test_wilkinson_like():
    """Clustered integer roots, a classic stress case."""
    true = list(range(1, 11))
    coeffs = poly_from_roots(true)
    res = chn.solve_all_chn(coeffs)
    assert match_error(res.roots, true) < 1e-6


if __name__ == "__main__":
    import sys
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
            traceback.print_exc(limit=2)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
