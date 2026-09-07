"""Robustness experiments: where the conditional hybrid actually earns its keep.

The timing sweep in :mod:`benchmark` uses random real-rooted polynomials, on
which Newton's method very rarely misbehaves.  The experiments here probe the
cases it does: attracting cycles, critical points, clustered and multiple
roots, and starting points far from any root.

Produces ``data/robustness.json`` and ``data/basins.npz``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import chn_solvers as chn

MASTER_SEED = 20260828
DATA_DIR = Path(__file__).parent / "data"

# Polynomials on which Newton's method is known to be badly behaved.
HARD_POLYNOMIALS: Dict[str, List[float]] = {
    # Smale's cycling example: Newton has an attracting 2-cycle {0, 1}.
    "smale_cycle": [1.0, 0.0, -2.0, 2.0],
    # The canonical polynomiography subject; Newton's basins are fractal.
    "z3_minus_1": [1.0, 0.0, 0.0, -1.0],
    # A triple root at 1: Newton degrades to linear convergence.
    "triple_root": [1.0, -3.0, 3.0, -1.0],
    # Two tight clusters at +-1, separated by 1e-3.
    "clustered": list(
        np.real(np.poly([1.0, 1.0 + 1e-3, 1.0 - 1e-3, -1.0, -1.0 + 1e-3]))
    ),
    # High-degree Chebyshev polynomial: many roots, all real, tightly packed.
    "chebyshev_20": list(np.polynomial.chebyshev.cheb2poly([0.0] * 20 + [1.0])[::-1]),
    # Wilkinson's polynomial of degree 12, the classic conditioning stress test.
    "wilkinson_12": list(np.real(np.poly(list(range(1, 13))))),
}


def _newton_only(coeffs: Sequence[complex], z0: complex, max_iter: int = 500):
    """Unguarded Newton iteration, for comparison against the hybrid."""
    z = complex(z0)
    for it in range(1, max_iter + 1):
        p, dp = chn.horner_with_derivative(coeffs, z)
        if abs(p) <= chn.backward_error_bound(coeffs, z):
            return z, it, True
        if abs(dp) < 1e-300:
            return z, it, False
        z = z - p / dp
        if abs(z) > 1e12 or not np.isfinite(abs(z)):
            return z, it, False
    return z, max_iter, False


def seed_sweep(trials: int = 4000, radius: float = 2.5) -> Dict:
    """From how many starting points does each method reach a root?

    For every hard polynomial a fixed set of starting points is drawn from a
    disk, and each single-root iteration is run from each of them.  Newton is
    counted as failing when it cycles, diverges, or exhausts its iteration
    budget.  The hybrid cannot do any of those things: its modulus sequence is
    strictly decreasing and bounded below.
    """
    out: Dict[str, Dict[str, float]] = {}
    rng = np.random.default_rng(MASTER_SEED)
    angles = rng.uniform(0, 2 * np.pi, trials)
    radii = radius * np.sqrt(rng.uniform(0, 1, trials))
    seeds = radii * np.exp(1j * angles)

    def classify(coeffs, z0, z_final, iterations, budget):
        """Converged, merely slow, or genuinely broken?

        The distinction matters.  An iteration that has driven ``|p|`` down by
        orders of magnitude and simply ran out of budget is behaving as the
        theory predicts, only slowly; one that ends no closer than it started,
        or that has run off to infinity, has failed outright.  Newton can do
        the latter -- it has attracting cycles -- while the hybrid cannot,
        because its modulus sequence is strictly decreasing.
        """
        if abs(chn.horner(coeffs, z_final)) <= 1e3 * chn.backward_error_bound(
            coeffs, z_final
        ):
            return "converged"
        start = abs(chn.horner(coeffs, z0))
        end = abs(chn.horner(coeffs, z_final))
        if not np.isfinite(end) or abs(z_final) > 1e12:
            return "diverged"
        if start > 0 and end < 1e-3 * start:
            return "stalled"
        return "diverged"

    for name, coeffs in HARD_POLYNOMIALS.items():
        tally = {k: {"converged": 0, "stalled": 0, "diverged": 0}
                 for k in ("newton", "chn", "chn_t")}
        iters = {"newton": [], "chn": [], "chn_t": []}
        for z0 in seeds:
            z, it, _ = _newton_only(coeffs, z0)
            verdict = classify(coeffs, z0, z, it, 500)
            tally["newton"][verdict] += 1
            if verdict == "converged":
                iters["newton"].append(it)
            for key, trunc in (("chn", False), ("chn_t", True)):
                res = chn.solve_chn(coeffs, z0, max_iter=500, truncated=trunc)
                verdict = classify(coeffs, z0, res.roots[0], res.iterations, 500)
                tally[key][verdict] += 1
                if verdict == "converged":
                    iters[key].append(res.iterations)

        out[name] = {"degree": len(coeffs) - 1, "trials": trials}
        for key in ("newton", "chn", "chn_t"):
            for verdict, count in tally[key].items():
                out[name][f"{key}_{verdict}_pct"] = 100.0 * count / trials
            out[name][f"{key}_median_iters"] = (
                float(np.median(iters[key])) if iters[key] else float("nan")
            )

        print(
            f"{name:14s} deg {out[name]['degree']:2d} | converged "
            f"Newton {out[name]['newton_converged_pct']:6.2f}%  "
            f"CHN {out[name]['chn_converged_pct']:6.2f}%  "
            f"CHN-T {out[name]['chn_t_converged_pct']:6.2f}%  | broken outright "
            f"{out[name]['newton_diverged_pct']:5.2f}% / "
            f"{out[name]['chn_diverged_pct']:5.2f}% / "
            f"{out[name]['chn_t_diverged_pct']:5.2f}%",
            flush=True,
        )
    return out


def stall_study(degrees: Sequence[int] = (5, 10, 15, 20, 25),
                trials: int = 60) -> Dict:
    """How often does the *guaranteed* RNM step stall, and does truncation fix it?

    Full RNM is guaranteed to decrease the modulus, but the guaranteed step is
    short: on a flat stretch of the modulus surface it can take hundreds of
    iterations to cross.  The adaptive truncated step takes the longest of the
    candidate lengths along the same direction that still decreases the
    modulus, and the full step remains available at ``m = n``, so nothing is
    lost.
    """
    out = {"degrees": list(degrees), "chn_capped_pct": [], "chn_t_capped_pct": [],
           "chn_rnm_steps": [], "chn_t_rnm_steps": [],
           "mean_truncation_order": [], "truncation_histogram": {}}
    order_pool: List[int] = []
    for degree in degrees:
        rng = np.random.default_rng(MASTER_SEED + degree)
        capped = {"chn": 0, "chn_t": 0}
        rnm = {"chn": [], "chn_t": []}
        for _ in range(trials):
            coeffs = list(np.real(np.poly(rng.uniform(-1, 1, degree))))
            for key, trunc in (("chn", False), ("chn_t", True)):
                res = chn.solve_all_chn(coeffs, truncated=trunc)
                capped[key] += 0 if res.converged else 1
                rnm[key].append(res.rnm_steps)
                if trunc:
                    order_pool.extend(res.truncation_orders)
        out["chn_capped_pct"].append(100.0 * capped["chn"] / trials)
        out["chn_t_capped_pct"].append(100.0 * capped["chn_t"] / trials)
        out["chn_rnm_steps"].append(float(np.mean(rnm["chn"])))
        out["chn_t_rnm_steps"].append(float(np.mean(rnm["chn_t"])))
        print(f"deg {degree:2d} | iteration cap hit: "
              f"CHN {out['chn_capped_pct'][-1]:5.1f}% "
              f"vs CHN-T {out['chn_t_capped_pct'][-1]:5.1f}% | mean RNM steps "
              f"{out['chn_rnm_steps'][-1]:7.1f} vs {out['chn_t_rnm_steps'][-1]:6.1f}",
              flush=True)
    if order_pool:
        values, counts = np.unique(order_pool, return_counts=True)
        out["truncation_histogram"] = {
            int(v): float(100.0 * c / len(order_pool)) for v, c in zip(values, counts)
        }
        out["mean_truncation_order"] = float(np.mean(order_pool))
        top = sorted(out["truncation_histogram"].items())[:5]
        print("\ntruncation order m used by CHN-T: "
              + ", ".join(f"m={k}: {v:.1f}%" for k, v in top))
    return out


def basin_grid(coeffs: Sequence[complex], method: str, roots: Sequence[complex],
               resolution: int = 420, limit: float = 2.0):
    """Basin-of-attraction and iteration-count rasters for polynomiography."""
    axis = np.linspace(-limit, limit, resolution)
    basin = np.full((resolution, resolution), -1, dtype=np.int16)
    iters = np.zeros((resolution, resolution), dtype=np.int16)

    for i, y in enumerate(axis):
        for j, x in enumerate(axis):
            z0 = complex(x, y)
            if method == "newton":
                z, it, ok = _newton_only(coeffs, z0, max_iter=120)
            else:
                res = chn.solve_chn(coeffs, z0, max_iter=120,
                                    truncated=(method == "chn_t"))
                z, it = res.roots[0], res.iterations
                ok = abs(chn.horner(coeffs, z)) <= 1e3 * chn.backward_error_bound(
                    coeffs, z)
            iters[i, j] = it
            if ok:
                basin[i, j] = int(np.argmin([abs(z - r) for r in roots]))
    return basin, iters


def build_basins() -> Dict[str, np.ndarray]:
    """Raster data for the polynomiographs used in the manuscript."""
    payload: Dict[str, np.ndarray] = {}
    targets = {
        "z3_minus_1": ([1.0, 0.0, 0.0, -1.0], 2.0),
        "smale_cycle": ([1.0, 0.0, -2.0, 2.0], 2.5),
    }
    for name, (coeffs, limit) in targets.items():
        roots = list(np.roots(coeffs))
        payload[f"{name}_roots"] = np.array(roots)
        for method in ("newton", "chn", "chn_t"):
            started = time.perf_counter()
            basin, iters = basin_grid(coeffs, method, roots, limit=limit)
            payload[f"{name}_{method}_basin"] = basin
            payload[f"{name}_{method}_iters"] = iters
            failures = int((basin < 0).sum())
            print(f"{name:12s} {method:7s} | failures {failures:6d} / {basin.size} "
                  f"({100.0 * failures / basin.size:5.2f}%) "
                  f"[{time.perf_counter() - started:.1f}s]", flush=True)
        payload[f"{name}_limit"] = np.array(limit)
    return payload


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    print("=== Experiment 1: convergence from arbitrary starting points ===")
    seeds = seed_sweep()

    print("\n=== Experiment 2: stalling of the guaranteed RNM step ===")
    stalls = stall_study()

    print("\n=== Experiment 3: basin rasters for polynomiography ===")
    basins = build_basins()

    (DATA_DIR / "robustness.json").write_text(
        json.dumps({"seed_sweep": seeds, "stall_study": stalls}, indent=1)
    )
    np.savez_compressed(DATA_DIR / "basins.npz", **basins)
    print(f"\nWrote {DATA_DIR / 'robustness.json'} and {DATA_DIR / 'basins.npz'}")


if __name__ == "__main__":
    main()
