"""Benchmark harness: CHN and CHN-T against Yuksel (2022) and baselines.

Produces ``data/benchmark.json`` consumed by :mod:`figures` and by the
manuscript.  Everything is seeded, so the numbers are reproducible.

Usage
-----
``python benchmark.py [--trials 100] [--max-degree 25]``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

import chn_solvers as chn

# Master seed. Every random draw in the study derives from this value.
MASTER_SEED = 20260828

# A true root counts as recovered when some returned root lies this close.
MATCH_TOL = 1e-6

DATA_DIR = Path(__file__).parent / "data"


# --------------------------------------------------------------------------- #
# Test-set construction
# --------------------------------------------------------------------------- #
def monic_from_roots(roots: Sequence[complex]) -> np.ndarray:
    """Monic polynomial with the prescribed roots, descending coefficients."""
    coeffs = np.array([1.0 + 0.0j])
    for r in roots:
        coeffs = np.convolve(coeffs, np.array([1.0 + 0.0j, -complex(r)]))
    return coeffs


def real_rooted_instance(rng: np.random.Generator, degree: int):
    """Degree-``degree`` monic polynomial with roots drawn from ``U(-1, 1)``.

    This is the family in which Yuksel's method is at its strongest: all roots
    are real and lie inside a known bounded interval, exactly the situation a
    ray tracer or a collision-detection query presents.
    """
    roots = rng.uniform(-1.0, 1.0, degree)
    coeffs = np.real(monic_from_roots(roots))
    return list(coeffs), list(roots)


def mixed_roots_instance(rng: np.random.Generator, degree: int):
    """Real coefficients, roots split between the real axis and the plane.

    Half the roots are real; the rest form conjugate pairs.  Interval methods
    can only ever report the real half.
    """
    n_complex_pairs = degree // 4
    n_real = degree - 2 * n_complex_pairs
    real_roots = list(rng.uniform(-1.0, 1.0, n_real))
    roots = list(real_roots)
    for _ in range(n_complex_pairs):
        z = complex(rng.uniform(-1.0, 1.0), rng.uniform(0.2, 1.0))
        roots.extend([z, z.conjugate()])
    coeffs = np.real(monic_from_roots(roots))
    return list(coeffs), roots, real_roots


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def forward_error(found: Sequence[complex], true: Sequence[complex]) -> float:
    """Largest distance from a true root to the nearest returned root."""
    if not len(found):
        return float("inf")
    return max(min(abs(complex(t) - complex(f)) for f in found) for t in true)


def recovery_rate(found: Sequence[complex], true: Sequence[complex]) -> float:
    """Fraction of true roots matched to within :data:`MATCH_TOL`."""
    if not len(found):
        return 0.0
    hits = sum(
        1 for t in true if min(abs(complex(t) - complex(f)) for f in found) <= MATCH_TOL
    )
    return hits / len(true)


def max_residual(coeffs: Sequence[complex], found: Sequence[complex]) -> float:
    """Largest ``|p(r)|`` over the returned roots, relative to the Adams bound.

    A value of order one means the roots are backward stable: each is an exact
    root of a polynomial within rounding error of the input.
    """
    if not len(found):
        return float("inf")
    out = 0.0
    for r in found:
        bound = chn.backward_error_bound(coeffs, r)
        out = max(out, abs(chn.horner(coeffs, r)) / max(bound, 1e-300))
    return out


def timed(fn: Callable[[], object], repeats: int = 3):
    """Best-of-``repeats`` wall-clock time in milliseconds, plus the result."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - start) * 1e3
        best = min(best, elapsed)
    return result, best


# --------------------------------------------------------------------------- #
# Main sweep
# --------------------------------------------------------------------------- #
def run_real_rooted_sweep(degrees: Sequence[int], trials: int) -> Dict:
    """Per-degree comparison on real-rooted polynomials."""
    methods = ["chn", "chn_t", "newton", "yuksel"]
    out: Dict[str, Dict[str, List[float]]] = {
        m: {k: [] for k in
            ("time_med", "time_mean", "evals", "err_med", "err_max",
             "recovery", "residual", "iters")}
        for m in methods
    }
    out["meta"] = {"degrees": list(degrees), "trials": trials,
                   "seed": MASTER_SEED, "match_tol": MATCH_TOL}
    speedups: Dict[str, List[float]] = {"chn": [], "chn_t": []}
    fallback_share: List[float] = []
    trunc_orders: List[float] = []

    for degree in degrees:
        rng = np.random.default_rng(MASTER_SEED + degree)
        per: Dict[str, Dict[str, List[float]]] = {
            m: {k: [] for k in ("t", "e", "err", "rec", "res", "it")}
            for m in methods
        }
        newton_share: List[float] = []
        orders_here: List[int] = []
        ratio_chn: List[float] = []
        ratio_chn_t: List[float] = []

        for _ in range(trials):
            coeffs, true_roots = real_rooted_instance(rng, degree)
            radius = chn.kalantari_bound(coeffs)

            runs = {
                "chn": lambda: chn.solve_all_chn(coeffs, truncated=False),
                "chn_t": lambda: chn.solve_all_chn(coeffs, truncated=True),
                "newton": lambda: chn.solve_newton_deflation(coeffs),
                "yuksel": lambda: chn.solve_yuksel(coeffs, -radius, radius),
            }
            times = {}
            for name, fn in runs.items():
                res, ms = timed(fn)
                times[name] = ms
                per[name]["t"].append(ms)
                per[name]["e"].append(res.evals.weighted_ops)
                per[name]["it"].append(res.iterations)
                per[name]["err"].append(forward_error(res.roots, true_roots))
                per[name]["rec"].append(recovery_rate(res.roots, true_roots))
                per[name]["res"].append(max_residual(coeffs, res.roots))
                if name == "chn":
                    total = res.newton_steps + res.rnm_steps
                    if total:
                        newton_share.append(res.rnm_steps / total)
                if name == "chn_t":
                    orders_here.extend(res.truncation_orders)

            ratio_chn.append(times["yuksel"] / max(times["chn"], 1e-12))
            ratio_chn_t.append(times["yuksel"] / max(times["chn_t"], 1e-12))

        for name in methods:
            d = per[name]
            out[name]["time_med"].append(float(np.median(d["t"])))
            out[name]["time_mean"].append(float(np.mean(d["t"])))
            out[name]["evals"].append(float(np.mean(d["e"])))
            out[name]["iters"].append(float(np.mean(d["it"])))
            finite = [v for v in d["err"] if np.isfinite(v)]
            out[name]["err_med"].append(
                float(np.median(finite)) if finite else float("nan"))
            out[name]["err_max"].append(
                float(np.max(finite)) if finite else float("nan"))
            out[name]["recovery"].append(float(np.mean(d["rec"])) * 100.0)
            resid = [v for v in d["res"] if np.isfinite(v)]
            out[name]["residual"].append(
                float(np.median(resid)) if resid else float("nan"))

        speedups["chn"].append(float(np.median(ratio_chn)))
        speedups["chn_t"].append(float(np.median(ratio_chn_t)))
        fallback_share.append(
            float(np.mean(newton_share)) * 100.0 if newton_share else 0.0)
        trunc_orders.append(
            float(np.mean(orders_here)) if orders_here else float("nan"))

        print(
            f"deg {degree:2d} | CHN {out['chn']['time_med'][-1]:7.3f} ms  "
            f"CHN-T {out['chn_t']['time_med'][-1]:7.3f} ms  "
            f"Yuksel {out['yuksel']['time_med'][-1]:7.3f} ms  "
            f"| speedup {speedups['chn'][-1]:5.2f}x / {speedups['chn_t'][-1]:5.2f}x "
            f"| CHN err {out['chn']['err_med'][-1]:.1e} "
            f"rec {out['chn']['recovery'][-1]:5.1f}%  "
            f"YK rec {out['yuksel']['recovery'][-1]:5.1f}% "
            f"| RNM {fallback_share[-1]:4.1f}% of steps",
            flush=True,
        )

    out["speedup_median"] = speedups
    out["rnm_fallback_pct"] = fallback_share
    out["mean_truncation_order"] = trunc_orders
    return out


def run_mixed_sweep(degrees: Sequence[int], trials: int) -> Dict:
    """Coverage experiment: which roots does each method actually return?"""
    out = {"degrees": list(degrees), "chn_all": [], "yuksel_all": [],
           "yuksel_real": [], "trials": trials}
    for degree in degrees:
        rng = np.random.default_rng(MASTER_SEED + 1000 + degree)
        chn_all, yk_all, yk_real = [], [], []
        for _ in range(trials):
            coeffs, all_roots, real_roots = mixed_roots_instance(rng, degree)
            radius = chn.kalantari_bound(coeffs)
            r_chn = chn.solve_all_chn(coeffs)
            r_yk = chn.solve_yuksel(coeffs, -radius, radius)
            chn_all.append(recovery_rate(r_chn.roots, all_roots))
            yk_all.append(recovery_rate(r_yk.roots, all_roots))
            yk_real.append(
                recovery_rate(r_yk.roots, real_roots) if real_roots else 1.0
            )
        out["chn_all"].append(float(np.mean(chn_all)) * 100.0)
        out["yuksel_all"].append(float(np.mean(yk_all)) * 100.0)
        out["yuksel_real"].append(float(np.mean(yk_real)) * 100.0)
        print(f"mixed deg {degree:2d} | CHN all-roots {out['chn_all'][-1]:5.1f}%  "
              f"Yuksel all-roots {out['yuksel_all'][-1]:5.1f}%  "
              f"(real subset {out['yuksel_real'][-1]:5.1f}%)", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--min-degree", type=int, default=3)
    parser.add_argument("--max-degree", type=int, default=25)
    args = parser.parse_args()

    degrees = list(range(args.min_degree, args.max_degree + 1))
    print(f"CHN benchmark | degrees {degrees[0]}-{degrees[-1]} | "
          f"{args.trials} polynomials per degree | seed {MASTER_SEED}\n")

    started = time.perf_counter()
    main_sweep = run_real_rooted_sweep(degrees, args.trials)
    print()
    mixed = run_mixed_sweep([d for d in degrees if d % 4 == 0 or d in (5, 25)],
                            max(args.trials // 4, 10))

    payload = {"real_rooted": main_sweep, "mixed": mixed}
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "benchmark.json").write_text(json.dumps(payload, indent=1))
    print(f"\nWrote {DATA_DIR / 'benchmark.json'} "
          f"in {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
