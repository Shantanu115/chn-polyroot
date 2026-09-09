# CHN-PolyRoot

A conditional hybrid of Newton's method and Kalantari's Robust Newton Method (RNM) for finding **all** roots of a polynomial, benchmarked against Yuksel's (2022) interval-splitting solver.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Idea

Take the Newton step whenever it decreases `|p(z)|`; otherwise take an RNM step, which decreases `|p(z)|` by construction. The modulus decreases strictly at every iteration, so — unlike plain Newton — **the iteration cannot cycle**.

## Results

100 random real-rooted polynomials per degree, roots drawn from `U(-1, 1)`. Reproduced by `CHN_polynomial_root_finding.ipynb`.

| Degree | CHN-T | Yuksel | Speedup | Forward error |
|:------:|------:|-------:|:-------:|:--------------:|
| 5      | 0.390 ms  | 0.873 ms  | 2.28×  | 1.1e-15 |
| 15     | 2.602 ms  | 12.490 ms | 4.86×  | 3.8e-12 |
| 25     | 7.326 ms  | 49.431 ms | 6.62×  | 1.5e-08 |

- Faster than Yuksel's method at every degree `n ≥ 4`, peaking at 6.62× at `n = 25`.
- Backward stable — forward error tracks the LAPACK companion-matrix eigensolver at every degree.
- On polynomials with complex roots, Yuksel returns about half the roots by construction; CHN and CHN-T return all of them.

<img src="figures/fig_performance.png" alt="Performance comparison" width="600">

On `z³ - 2z + 2`, where Newton has an attracting 2-cycle, plain Newton fails to converge from 0.76% of a 420×420 grid of starting points. CHN-T fails from none of them.

<img src="figures/fig_basins_smale.png" alt="Basins of attraction" width="600">

## Quick start

```bash
git clone https://github.com/Shantanu115/chn-polyroot
cd chn-polyroot
pip install -r requirements.txt

python test_chn_solvers.py     # 11 correctness tests, ~2 min
python benchmark.py            # timing/accuracy sweep -> data/benchmark.json
python experiments.py          # robustness & basins    -> data/robustness.json, basins.npz
python figures.py              # figures                -> figures/*.pdf, *.png
```

Or open `CHN_polynomial_root_finding.ipynb`, which reproduces every number and figure above, top to bottom, in about five minutes.

## Usage

```python
import chn_solvers as chn

coeffs = [1.0, 0.0, -2.0, 2.0]   # z^3 - 2z + 2, descending order
result = chn.solve_all_chn(coeffs, truncated=True)

result.roots        # all 3 roots, complex included
result.iterations, result.rnm_steps
```

Coefficients are always in **descending** order, matching `numpy.roots`.

| Function | Description |
|---|---|
| `solve_all_chn(coeffs, truncated=True)` | All `n` roots — recommended entry point |
| `solve_chn(coeffs, z0)` | Single root from one starting point |
| `rnm_step(coeffs, z)` | RNM increment, closed form |
| `truncated_rnm_step(...)` | Adaptive truncated RNM (smallest `m` that descends) |
| `solve_yuksel(coeffs, lo, hi)` | Baseline: real roots in an interval |
| `solve_newton_deflation(coeffs)` | Ablation: Newton with no robust fallback |
| `kalantari_bound(coeffs)` | Bound `U_2` on the modulus of the roots |

## Repository layout

```
chn_solvers.py        solvers (standard library only)
benchmark.py           timing/accuracy sweep, degrees 3-25
experiments.py          robustness, stalling, basin rasters
figures.py                publication figures (PDF + 300 DPI PNG)
test_chn_solvers.py       correctness tests
build_notebook.py          regenerates the notebook from source
paper/                      manuscript + bibliography
data/                        benchmark output (JSON, NPZ)
figures/                      generated figures
```

All random draws derive from the master seed `20260828` in `benchmark.py`. Timing is best-of-three wall clock per instance, reported as the median over 100 instances — absolute times depend on the machine, ratios do not.

## References

- B. Kalantari, *Polynomial Root-Finding and Polynomiography*, World Scientific, 2008.
- B. Kalantari, "A geometric modulus principle for polynomials," *Amer. Math. Monthly* 118 (2011), 931–935.
- C. Yuksel, "High-performance polynomial root finding for graphics," *Proc. ACM Comput. Graph. Interact. Tech.* 5(3), Art. 7, 2022.

## License

[MIT](LICENSE)
