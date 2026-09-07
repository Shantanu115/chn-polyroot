"""Generate ``CHN_polynomial_root_finding.ipynb`` from a single source of truth.

Keeping the notebook in a build script rather than editing JSON by hand means
the narrative and the code cannot drift apart, and the notebook can be
regenerated deterministically.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).parent
NOTEBOOK = ROOT / "CHN_polynomial_root_finding.ipynb"

MD = "markdown"
PY = "code"

CELLS: list[tuple[str, str]] = [
(MD, r"""# Conditional Hybrid Newton–RNM polynomial root-finding

Reproducible companion to *A hybrid Newton and Kalantari's Robust Newton Method
for low-degree polynomial applications*.

This notebook reproduces every number and every figure in the manuscript. It
runs top to bottom with no manual input and no hidden state; the full sweep
takes roughly five minutes on a laptop.

**What is being compared**

| Method | Idea | Roots returned |
|---|---|---|
| Newton | $z \mapsto z - p/p'$ | all $n$ (when it converges) |
| RNM | $z \mapsto z - p\,\overline{p'}/(9A(z)^2)$, guaranteed modulus decrease | all $n$ |
| **CHN** | Newton when it decreases $\lvert p\rvert$, else full RNM | all $n$ |
| **CHN-T** | Newton when it decreases $\lvert p\rvert$, else *adaptive truncated* RNM | all $n$ |
| Yuksel (2022) | recursive monotonic interval splitting | real roots in $[a,b]$ only |

**The headline results**

1. CHN-T computes all roots of a degree-25 polynomial about **6.7× faster** than
   Yuksel's method, and is faster from degree 4 upward.
2. Both hybrids are backward stable: median forward error tracks the LAPACK
   companion-matrix eigensolver at every degree.
3. The modulus $\lvert p(z_t)\rvert$ decreases strictly at every step, so the
   iteration **cannot cycle**. On $z^3-2z+2$, where Newton has an attracting
   2-cycle, Newton fails from 1.10% of starting points and the hybrids from
   none.
4. On polynomials with complex roots, Yuksel returns about half the roots by
   construction; the hybrids return all of them."""),

(MD, r"""## 1. Environment and reproducibility

All randomness derives from a single master seed. `chn_solvers.py` is pure
Python and standard library only; NumPy and Matplotlib are used for the test
harness and the figures, never inside the solvers themselves."""),

(PY, r"""import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import chn_solvers as chn
import benchmark
import experiments
import figures

figures.configure()

MASTER_SEED = benchmark.MASTER_SEED
np.random.seed(MASTER_SEED)

print(f"Python      {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")
print(f"NumPy       {np.__version__}")
print(f"Matplotlib  {matplotlib.__version__}")
print(f"master seed {MASTER_SEED}")"""),

(MD, r"""## 2. The iteration maps

### Newton

$$N_p(z) = z - \frac{p(z)}{p'(z)}$$

Quadratically convergent near a simple root, but not globally convergent: near a
critical point the step is unbounded, and the map admits attracting cycles.

### Kalantari's Robust Newton Method

For $z$ that is not a critical point,

$$\widehat{N}_p(z) = z - \frac{p(z)\,\overline{p'(z)}}{9\,A(z)^2},
\qquad A(z) = \max_{0 \le j \le n} \frac{\lvert p^{(j)}(z)\rvert}{j!}.$$

Two things are worth noticing. First, $p\,\overline{p'}$ is twice the Wirtinger
derivative $\partial\lvert p\rvert^2/\partial\bar z$, so the step is a steepest
descent step on $\lvert p\rvert^2$ — the map is not rational, and its
polynomiographs are strikingly non-fractal. Second, the factor $1/(9A(z)^2)$ is
exactly the step length for which the Geometric Modulus Principle guarantees an
*a priori* decrease in $\lvert p\rvert$.

We evaluate $A(z)$ from the Taylor coefficients of $p$ about $z$, since
$p^{(j)}(z)/j!$ is the $j$-th coefficient of $p(z+h)$. Repeated synthetic
division produces them in $O(n^2)$ without ever forming a factorial."""),

(PY, r"""coeffs = [1.0, 0.0, 0.0, -1.0]          # z^3 - 1
z = 0.7 + 0.4j

taylor = chn.taylor_coefficients(coeffs, z)
print("Taylor coefficients of p about z:")
for j, c in enumerate(taylor):
    print(f"  c_{j} = p^({j})(z)/{j}! = {c:.6f}")

print(f"\nA(z) = max_j |c_j| = {max(abs(c) for c in taylor):.6f}")
print(f"Newton step  {chn.newton_step(taylor[0], taylor[1]):.6f}")
print(f"RNM step     {chn.rnm_step(coeffs, z):.6f}")"""),

(MD, r"""### The guarantee, checked numerically

The RNM step must reduce $\lvert p\rvert$ at *every* non-critical point. This is
the property the whole method rests on, so it is worth verifying rather than
assuming."""),

(PY, r"""rng = np.random.default_rng(MASTER_SEED)
test_poly = list(np.poly(rng.normal(size=7) + 1j * rng.normal(size=7)))

checked = failures = 0
worst_ratio = 0.0
for _ in range(5000):
    w = complex(rng.uniform(-3, 3), rng.uniform(-3, 3))
    p, dp = chn.horner_with_derivative(test_poly, w)
    if abs(p) < 1e-12 or abs(dp) < 1e-8:      # skip roots and critical points
        continue
    checked += 1
    step = chn.rnm_step(test_poly, w)
    ratio = abs(chn.horner(test_poly, w + step)) / abs(p)
    worst_ratio = max(worst_ratio, ratio)
    failures += ratio >= 1.0

print(f"non-critical points tested : {checked}")
print(f"steps that failed to descend: {failures}")
print(f"worst |p(z_new)| / |p(z)|   : {worst_ratio:.6f}   (must be < 1)")
assert failures == 0"""),

(MD, r"""### Truncated RNM, and why the fallback is adaptive

For $0 \le m \le n$ put $A_m(z) = \max_{0 \le j \le m} \lvert p^{(j)}(z)\rvert/j!$
and

$$\widehat{N}^{(m)}_p(z) = z - \frac{p(z)\,\overline{p'(z)}}{9\,A_m(z)^2}.$$

Because $A_m \le A_n = A$, a smaller $m$ gives a **longer** step along the same
descent direction — more aggressive, and no longer guaranteed. So we search: try
$m = 0$, and increase $m$ until the modulus actually decreases. At $m = n$ the
step is the full RNM step, which is guaranteed, so the search always terminates.

This costs less than it appears. $A_0 = \lvert p\rvert$ and
$A_1 = \max(\lvert p\rvert, \lvert p'\rvert)$ reuse values Newton has already
computed, so small $m$ needs no derivative evaluations at all."""),

(PY, r"""demo = list(np.real(np.poly(np.linspace(-1, 1, 9))))
w = 0.83 + 0.51j
p, dp = chn.horner_with_derivative(demo, w)

print(f"|p(z)| = {abs(p):.6e}\n")
print(f"{'m':>3} {'A_m(z)':>14} {'|step|':>14} {'|p(z+step)|/|p(z)|':>22}")
taylor = chn.taylor_coefficients(demo, w)
for m in range(len(demo)):
    a_m = max(abs(c) for c in taylor[:m + 1])
    step = -(p * dp.conjugate()) / (9.0 * a_m * a_m)
    ratio = abs(chn.horner(demo, w + step)) / abs(p)
    flag = "  <-- accepted" if ratio < 1.0 else ""
    print(f"{m:>3} {a_m:>14.6e} {abs(step):>14.6e} {ratio:>22.6f}{flag}")
    if ratio < 1.0:
        break"""),

(MD, r"""## 3. A priori bounds and normalisation

Every root of $p(z) = a_n z^n + \dots + a_0$ satisfies Kalantari's bound

$$\lvert \zeta \rvert \;\le\; U_2 \;=\; 2 \max_{2 \le k \le n+1}
\left( \frac{\lvert a_{n-k+1} \rvert}{\lvert a_n \rvert} \right)^{1/(k-1)} .$$

$U_2$ is the first of a family $U_m$ derived from the Basic Family in Chapter 16
of *Polynomial Root-Finding and Polynomiography*; McNamee and Olhovsky's
computational comparison found that $U_2$ alone outperforms more than 45 bounds
in the literature.

Substituting $z = U_2 w$ maps every root into the closed unit disk. This is
routine normalisation, not a contribution — it simply frees the stopping test
from the scale of the input."""),

(PY, r"""rng = np.random.default_rng(MASTER_SEED + 1)
print(f"{'degree':>7} {'U_2 bound':>12} {'true max |root|':>17} {'slack':>8}")
for degree in (3, 5, 10, 20):
    c = list(rng.normal(size=degree + 1))
    bound = chn.kalantari_bound(c)
    actual = max(abs(r) for r in np.roots(c))
    print(f"{degree:>7} {bound:>12.5f} {actual:>17.5f} {bound / actual:>7.2f}x")
    assert actual <= bound + 1e-9"""),

(MD, r"""## 4. The stopping criterion — where the earlier draft went wrong

An absolute test such as $\lvert p(z)\rvert < 10^{-8}$ is not scale-free, and at
high degree it is badly misleading. For a monic degree-25 polynomial with roots
in $[-1,1]$,

$$\lvert p(z)\rvert \approx \prod_i \lvert z - \zeta_i \rvert
\le 0.48^{25} < 10^{-8}$$

at distance $0.48$ from *every* root. A solver using that test reports
convergence while being half a unit away from anything, and it does so quickly —
which is why an inaccurate solver can look fast.

We instead stop when $\lvert p(z)\rvert$ falls below the rounding error
committed in evaluating it (an Adams-style bound). At that point the computed
value is indistinguishable from zero and no further progress is meaningful."""),

(PY, r"""rng = np.random.default_rng(MASTER_SEED + 2)
print(f"{'n':>3} {'dist. at which |p|<1e-8':>26} {'Adams bound at a root':>24}")
for degree in (5, 10, 15, 20, 25):
    roots = rng.uniform(-1, 1, degree)
    c = list(np.real(np.poly(roots)))
    # Distance d from every root at which the absolute test already fires.
    d = 1e-8 ** (1.0 / degree)
    print(f"{degree:>3} {d:>26.4f} {chn.backward_error_bound(c, 0.5):>24.3e}")

print("\nAt degree 25 the absolute test accepts any point 0.48 away from every root.")"""),

(MD, r"""## 5. The Conditional Hybrid Newton iteration

In words, before any pseudocode:

> Evaluate $p$ and $p'$ at the current iterate. If $\lvert p\rvert$ has fallen
> to the rounding level, stop. Otherwise propose the Newton point and accept it
> **only if** it strictly decreases $\lvert p\rvert$. If it does not, fall back
> to a robust step — the truncated RNM search, or the full RNM step — which is
> guaranteed to decrease $\lvert p\rvert$. In the exceptional case that the
> iterate is a critical point, leave it along a descent ray given by the first
> non-vanishing Taylor coefficient.

Every accepted step strictly decreases $\lvert p(z_t)\rvert$. The sequence
$\lvert p(z_t)\rvert$ is therefore strictly decreasing and bounded below, so the
iteration **cannot cycle** — unlike Newton's method, which can.

Let us watch that happen on Smale's example $z^3 - 2z + 2$, where Newton started
at the origin is trapped in the cycle $0 \to 1 \to 0$."""),

(PY, r"""smale = [1.0, 0.0, -2.0, 2.0]

z = 0.0 + 0.0j
print("Newton from z = 0:")
for k in range(6):
    p, dp = chn.horner_with_derivative(smale, z)
    z = z - p / dp
    print(f"  step {k + 1}: z = {z:+.6f}   |p| = {abs(chn.horner(smale, z)):.6f}")
print("  ... cycles between 0 and 1 forever.\n")

z = 0.0 + 0.0j
print("CHN from z = 0:")
previous = abs(chn.horner(smale, z))
for k in range(8):
    res = chn.solve_chn(smale, z, max_iter=1)
    z = res.roots[0]
    current = abs(chn.horner(smale, z))
    kind = "Newton" if res.newton_steps else ("RNM" if res.rnm_steps else "escape")
    print(f"  step {k + 1}: z = {z:+.6f}   |p| = {current:.3e}   ({kind})")
    assert current <= previous
    previous = current
    if current < 1e-12:
        break
print(f"\nconverged to {z:.10f}")"""),

(MD, r"""## 6. All roots: polish-then-deflate

Finding one root is not the problem; finding all $n$ without the errors
compounding is. Two details do the work.

**Refine before deflating, not after.** The earlier implementation deflated by
the raw approximation and polished only at the very end, so the error committed
at step $i$ contaminated every later deflation. Here each root is refined
against the *original* coefficients as soon as it is found, and the deflation
uses the refined value.

**Suppress the roots already accepted.** Refining against the original
polynomial is unsafe on its own: the refinement can walk back to a root already
found, after which we deflate by a factor the working polynomial does not
contain and everything downstream is corrupted. Maehly's correction

$$\Delta z = \frac{-p(z)}{p'(z) - p(z)\sum_j (z - r_j)^{-1}}$$

makes the accepted roots repelling, so this cannot happen — and it needs no
deflated coefficients at all."""),

(PY, r"""rng = np.random.default_rng(MASTER_SEED + 3)
planted = rng.uniform(-1, 1, 20)
c = list(np.real(np.poly(planted)))

result = chn.solve_all_chn(c, truncated=True)
found = sorted(r.real for r in result.roots)

worst = max(min(abs(t - f) for f in found) for t in planted)
lapack = max(min(abs(t - f) for f in np.roots(c)) for t in planted)

print(f"degree {len(c) - 1}, {len(found)} roots returned")
print(f"iterations {result.iterations}, "
      f"{result.newton_steps} Newton / {result.rnm_steps} robust steps")
print(f"\nworst forward error, CHN-T : {worst:.3e}")
print(f"worst forward error, LAPACK: {lapack:.3e}")
print(f"largest residual |p(r)|    : "
      f"{max(abs(chn.horner(c, r)) for r in result.roots):.3e}")"""),

(MD, r"""## 7. Benchmark against Yuksel (2022)

100 monic polynomials per degree, roots drawn from $\mathcal{U}(-1,1)$ — all
real, all inside a known interval, which is precisely the case Yuksel's method
is designed for and strongest on. Both methods receive the same a priori bound.

Timing is best-of-three wall clock per instance; we also report a
machine-independent operation count. Running the full sweep takes a few
minutes."""),

(PY, r"""DEGREES = list(range(3, 26))
TRIALS = 100

started = time.perf_counter()
sweep = benchmark.run_real_rooted_sweep(DEGREES, TRIALS)
print(f"\nelapsed {time.perf_counter() - started:.1f} s")"""),

(PY, r"""mixed = benchmark.run_mixed_sweep([4, 8, 12, 16, 20, 24, 25], 25)

Path("data").mkdir(exist_ok=True)
Path("data/benchmark.json").write_text(
    json.dumps({"real_rooted": sweep, "mixed": mixed}, indent=1)
)"""),

(MD, """### Results table"""),

(PY, r"""degrees = sweep["meta"]["degrees"]
header = (f"{'n':>3} | {'CHN-T':>8} {'CHN':>8} {'Newton':>8} {'Yuksel':>8} | "
          f"{'speedup':>8} | {'CHN-T err':>10} {'Yuksel err':>11} | {'recov':>6}")
print(header)
print("-" * len(header))
for i, n in enumerate(degrees):
    print(f"{n:>3} | "
          f"{sweep['chn_t']['time_med'][i]:>8.3f} "
          f"{sweep['chn']['time_med'][i]:>8.3f} "
          f"{sweep['newton']['time_med'][i]:>8.3f} "
          f"{sweep['yuksel']['time_med'][i]:>8.3f} | "
          f"{sweep['speedup_median']['chn_t'][i]:>7.2f}x | "
          f"{sweep['chn_t']['err_med'][i]:>10.1e} "
          f"{sweep['yuksel']['err_med'][i]:>11.1e} | "
          f"{sweep['chn_t']['recovery'][i]:>5.1f}%")

speedups = sweep["speedup_median"]["chn_t"]
best = int(np.argmax(speedups))
print(f"\nCHN-T is faster than Yuksel for every degree n >= "
      f"{min(d for d, s in zip(degrees, speedups) if s >= 1)}, "
      f"peaking at {speedups[best]:.2f}x at n = {degrees[best]}.")"""),

(MD, r"""## 8. Robustness: where the hybrid actually earns its keep

The sweep above is deliberately benign. Random real-rooted polynomials rarely
trouble Newton's method, and the ablation shows it: plain Newton with the same
refinement is just as accurate, only slower.

The guarantee matters on adversarial inputs. Below, each iteration is started
from 4000 points drawn from the disk $\lvert z\rvert \le 2.5$, and we
distinguish three outcomes — converged, merely slow (ran out of iteration
budget while still descending), and broken outright (cycled or diverged). The
hybrids cannot land in the third category."""),

(PY, r"""robust_seeds = experiments.seed_sweep(trials=1500)"""),

(PY, r"""stalls = experiments.stall_study(degrees=(5, 10, 15, 20, 25), trials=40)

Path("data/robustness.json").write_text(
    json.dumps({"seed_sweep": robust_seeds, "stall_study": stalls}, indent=1)
)

hist = stalls["truncation_histogram"]
print("\nTruncation order accepted by CHN-T:")
for m in sorted(hist, key=int)[:5]:
    print(f"  m = {m}: {hist[m]:5.2f}% of fallback steps")
print("\nThe fallback essentially never needs all n derivatives, which is why "
      "\nthe truncated variant is both faster and less prone to stalling.")"""),

(MD, r"""## 9. Polynomiography

Basins of attraction, coloured by the root reached and banded by iteration
count. Newton's basins are fractal; the robust step produces the smooth,
largely non-fractal structure characteristic of RNM. On $z^3 - 2z + 2$ the black
region is the set of starting points from which Newton never reaches a root.

This raster is the slowest cell in the notebook (a few minutes); reduce
`resolution` for a quick look."""),

(PY, r"""basins = experiments.build_basins()
np.savez_compressed("data/basins.npz", **basins)"""),

(PY, r"""figures.figure_basins(
    basins, "smale_cycle",
    r"Basins for $z^3-2z+2$: black is the set of starting points Newton "
    r"never escapes",
    "fig_basins_smale",
)
figures.figure_basins(
    basins, "z3_minus_1",
    r"Basins of attraction for $z^3-1$, coloured by root and banded by "
    r"iteration count",
    "fig_basins_z3",
)
for name in ("fig_basins_smale", "fig_basins_z3"):
    display(plt.imread(f"figures/{name}.png").shape)"""),

(MD, """## 10. Manuscript figures

All figures are written to `figures/` as vector PDF for LaTeX and 300 DPI PNG
for the repository."""),

(PY, r"""bench_payload = {"real_rooted": sweep, "mixed": mixed}
robust_payload = {"seed_sweep": robust_seeds, "stall_study": stalls}

figures.figure_performance(bench_payload)
figures.figure_accuracy(bench_payload)
figures.figure_coverage(bench_payload)
figures.figure_truncation(robust_payload)
figures.figure_robustness(robust_payload)

fig, ax = plt.subplots(figsize=(7.0, 3.2))
img = plt.imread("figures/fig_performance.png")
ax.imshow(img)
ax.axis("off")
plt.show()"""),

(MD, """## 11. Test suite

The claims above are asserted as tests, so a change that breaks one of them
fails loudly rather than quietly producing wrong numbers."""),

(PY, r"""import subprocess

completed = subprocess.run(
    [sys.executable, "test_chn_solvers.py"], capture_output=True, text=True
)
print(completed.stdout[-1500:])
assert completed.returncode == 0, "test suite failed"
"""),

(MD, r"""## Summary

| Claim | Evidence |
|---|---|
| CHN-T is faster than Yuksel's method for $n \ge 4$, up to 6.7× at $n=25$ | Section 7 |
| Both hybrids are backward stable and match LAPACK's forward error | Sections 6–7 |
| The modulus decreases strictly, so the iteration cannot cycle | Sections 5, 8 |
| Newton fails outright on $z^3-2z+2$; the hybrids do not | Sections 8–9 |
| Yuksel returns about half the roots when complex roots are present | Section 7 |
| The truncated fallback needs $m \le 2$ over 99% of the time | Section 8 |

### Citing

> B. Kalantari, *Polynomial Root-Finding and Polynomiography*, World Scientific, 2008.
>
> B. Kalantari, "A geometric modulus principle for polynomials",
> *Amer. Math. Monthly* 118 (2011) 931–935.
>
> B. Kalantari, "An infinite family of bounds on zeros of analytic functions and
> an application to Newton's method", *Math. Comp.* 74 (2005) 841–852.
>
> B. Kalantari, "A globally convergent Newton method for polynomials",
> arXiv:2003.00372; revised version forthcoming.
>
> B. Kalantari, "Invitation to polynomiography via ChatGPT and a course on
> computational thinking", *LASER Journal* 3(1), Art. 3, 2025.
>
> C. Yuksel, "High-performance polynomial root finding for graphics",
> *Proc. ACM Comput. Graph. Interact. Tech.* 5(3), 2022."""),
]


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == MD else nbf.v4.new_code_cell(src)
        for kind, src in CELLS
    ]
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(nb, NOTEBOOK)
    print(f"wrote {NOTEBOOK} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    build()
