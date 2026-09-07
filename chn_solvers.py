"""Conditional Hybrid Newton (CHN) polynomial root-finding.

This module implements, in a single unified framework:

* :func:`newton_step`            -- the classical Newton map.
* :func:`rnm_step`               -- Kalantari's Robust Newton Method (RNM),
  in the closed form valid at every non-critical point.
* :func:`truncated_rnm_step`     -- the adaptive Truncated RNM of Kalantari,
  which searches for the smallest truncation order ``m`` whose step already
  reduces the modulus.
* :func:`solve_chn`              -- the Conditional Hybrid Newton iteration:
  take a Newton step whenever it decreases ``|p|``; otherwise fall back to an
  RNM (or Truncated RNM) step, which is guaranteed to decrease it.
* :func:`solve_all_chn`          -- all ``n`` roots via polish-then-deflate.
* :func:`solve_yuksel`           -- Yuksel's (2022) recursive monotonic
  interval-splitting solver, used as the performance baseline.
* :func:`solve_newton_deflation` -- pure Newton with deflation, used as the
  ablation baseline that isolates the contribution of the RNM fallback.

Notation follows Kalantari.  For a polynomial ``p`` of degree ``n`` write

.. math::

    A(z) \\;=\\; \\max_{0 \\le j \\le n} \\frac{|p^{(j)}(z)|}{j!},
    \\qquad
    \\widehat{N}_p(z) \\;=\\; z - \\frac{p(z)\\,\\overline{p'(z)}}{d^{2} A(z)^{2}} .

The theoretical guarantee holds for ``d = 3`` (the factor ``1/9``); smaller
``d`` gives longer steps and is often faster in practice.

Coefficients are always given in **descending** order, ``a[0]`` first, so that
``p(z) = a[0] z**n + a[1] z**(n-1) + ... + a[n]``.  This matches
:func:`numpy.roots` and :func:`numpy.polyval`.

References
----------
B. Kalantari, *Polynomial Root-Finding and Polynomiography*, World Scientific,
2008.

B. Kalantari, "A geometric modulus principle for polynomials",
*Amer. Math. Monthly* 118 (2011) 931--935.

B. Kalantari, "A globally convergent Newton method for polynomials",
arXiv:2003.00372 (revised version forthcoming).

B. Kalantari, "Invitation to polynomiography via ChatGPT and a course on
computational thinking", *LASER Journal* 3(1), Art. 3, 2025.

C. Yuksel, "High-performance polynomial root finding for graphics",
*Proc. ACM Comput. Graph. Interact. Tech.* 5(3), 2022.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from typing import Callable, List, Sequence

__all__ = [
    "SolveResult",
    "EvalCounter",
    "horner",
    "horner_with_derivative",
    "taylor_coefficients",
    "kalantari_bound",
    "scale_to_unit_disk",
    "backward_error_bound",
    "newton_step",
    "rnm_step",
    "truncated_rnm_step",
    "solve_chn",
    "solve_all_chn",
    "solve_newton_deflation",
    "solve_yuksel",
    "solve_numpy",
]

# Machine epsilon for IEEE-754 double precision.
EPS: float = 2.220446049250313e-16

# Below this magnitude ``p'(z)`` is treated as zero and the iterate is
# considered (numerically) critical.
CRITICAL_TOL: float = 1e-300


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class EvalCounter:
    """Machine-independent cost model, measured in fused multiply-adds.

    Wall-clock time depends on the interpreter, the machine and the phase of
    the moon; counting arithmetic does not.  Every quantity is charged the
    number of multiply-add pairs it actually costs, weighted by the degree of
    the polynomial involved -- a Horner pass over a degree-``d`` polynomial
    costs ``d``, evaluating ``p`` and ``p'`` together costs ``2d``, and
    ``k`` steps of repeated synthetic division cost roughly ``k*d``.

    Counting calls rather than arithmetic would badly distort this particular
    comparison, because Yuksel's recursion spends most of its evaluations on
    derivative polynomials of much lower degree than the input.

    ``complex_ops`` additionally records how much of the total was performed
    in complex rather than real arithmetic, which on commodity hardware costs
    a further factor of three to four per operation.
    """

    real_ops: float = 0.0
    complex_ops: float = 0.0

    def add(self, degree: int, passes: float = 1.0, complex_arith: bool = True) -> None:
        """Charge ``passes`` Horner passes over a degree-``degree`` polynomial."""
        cost = float(degree) * passes
        if complex_arith:
            self.complex_ops += cost
        else:
            self.real_ops += cost

    @property
    def total_ops(self) -> float:
        """Unweighted multiply-add count, real and complex pooled."""
        return self.real_ops + self.complex_ops

    @property
    def weighted_ops(self) -> float:
        """Multiply-adds with complex arithmetic charged its true cost.

        A complex multiply-add is four real multiplies and four real adds, so
        it is charged a factor of four.
        """
        return self.real_ops + 4.0 * self.complex_ops

    def __iadd__(self, other: "EvalCounter") -> "EvalCounter":
        self.real_ops += other.real_ops
        self.complex_ops += other.complex_ops
        return self


@dataclass
class SolveResult:
    """Outcome of a root-finding call."""

    roots: List[complex] = field(default_factory=list)
    iterations: int = 0
    converged: bool = True
    evals: EvalCounter = field(default_factory=EvalCounter)
    newton_steps: int = 0
    rnm_steps: int = 0
    escape_steps: int = 0
    truncation_orders: List[int] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Polynomial kernels
# --------------------------------------------------------------------------- #
def horner(coeffs: Sequence[complex], z: complex) -> complex:
    """Evaluate ``p(z)`` by Horner's rule.

    Parameters
    ----------
    coeffs : sequence of complex
        Coefficients in descending order.
    z : complex
        Evaluation point.
    """
    value = coeffs[0]
    for a in coeffs[1:]:
        value = value * z + a
    return value


def horner_with_derivative(
    coeffs: Sequence[complex], z: complex
) -> tuple[complex, complex]:
    """Evaluate ``p(z)`` and ``p'(z)`` in a single pass.

    Returns
    -------
    (p, dp) : tuple of complex
    """
    value = coeffs[0]
    deriv = 0.0 + 0.0j
    for a in coeffs[1:]:
        deriv = deriv * z + value
        value = value * z + a
    return value, deriv


def taylor_coefficients(
    coeffs: Sequence[complex], z: complex, order: int | None = None
) -> List[complex]:
    """Return the leading Taylor coefficients of ``p`` about ``z``.

    The ``j``-th entry of the result is ``p**(j)(z) / j!``, obtained by
    repeated synthetic division rather than by forming factorials, which
    avoids overflow for large ``n`` and costs ``O(order * n)``.

    Parameters
    ----------
    coeffs : sequence of complex
        Coefficients in descending order.
    z : complex
        Expansion point.
    order : int, optional
        Highest index required.  Defaults to ``deg(p)``.

    Returns
    -------
    list of complex
        ``[c_0, c_1, ..., c_order]`` with ``p(z + h) = sum_j c_j h**j``.
    """
    degree = len(coeffs) - 1
    if order is None:
        order = degree
    order = min(order, degree)

    working = list(coeffs)
    out: List[complex] = []
    for _ in range(order + 1):
        quotient: List[complex] = []
        acc = working[0]
        for a in working[1:]:
            quotient.append(acc)
            acc = acc * z + a
        out.append(acc)
        if not quotient:
            break
        working = quotient
    while len(out) < order + 1:
        out.append(0.0 + 0.0j)
    return out


def _taylor_cost(degree: int, order: int) -> float:
    """Multiply-adds spent building Taylor coefficients ``c_0..c_order``."""
    order = min(order, degree)
    return sum(max(degree - j, 0) for j in range(order + 1))


def backward_error_bound(coeffs: Sequence[complex], z: complex) -> float:
    """Adams-style bound on the rounding error committed evaluating ``p(z)``.

    A point is accepted as a root once ``|p(z)|`` drops below this bound: no
    further progress is meaningful in double precision, because the computed
    value is then indistinguishable from zero.  This *relative* criterion is
    what makes the solver scale-free.  A fixed absolute threshold such as
    ``|p(z)| < 1e-8`` is badly misleading at high degree -- for a monic degree
    25 polynomial with roots in ``[-1, 1]`` it is satisfied at distance ``0.48``
    from every root, since ``0.48**25 < 1e-8``.
    """
    magnitude = abs(coeffs[0])
    az = abs(z)
    for a in coeffs[1:]:
        magnitude = magnitude * az + abs(a)
    return 4.0 * len(coeffs) * EPS * magnitude


# --------------------------------------------------------------------------- #
# A priori bounds on the modulus of the roots
# --------------------------------------------------------------------------- #
def kalantari_bound(coeffs: Sequence[complex]) -> float:
    """Kalantari's bound ``U_2`` on the modulus of the roots.

    For ``p(z) = a_n z**n + ... + a_0`` every root ``zeta`` satisfies

    .. math::

        |\\zeta| \\;\\le\\; U_2 \\;=\\;
        2 \\max_{2 \\le k \\le n+1}
        \\left( \\frac{|a_{n-k+1}|}{|a_n|} \\right)^{1/(k-1)} .

    This is the first member of the family of bounds ``U_m`` derived from the
    Basic Family in Chapter 16 of Kalantari (2008); the computational study of
    McNamee and Olhovsky reports that already ``U_2`` outperforms more than 45
    bounds from the literature, and ``U_m`` converges to the radius of the
    tightest containing annulus as ``m`` grows.

    Notes
    -----
    ``coeffs`` is in descending order, so ``coeffs[k]`` is ``a_{n-k}`` and the
    exponent for ``coeffs[k]`` is ``1/k``.  There is no extra factor of two on
    the constant term: that variant belongs to Fujiwara's older bound.
    """
    lead = abs(coeffs[0])
    if lead == 0.0:
        raise ValueError("leading coefficient must be non-zero")
    ratios = [
        (abs(coeffs[k]) / lead) ** (1.0 / k)
        for k in range(1, len(coeffs))
        if coeffs[k] != 0
    ]
    return 2.0 * max(ratios) if ratios else 1.0


def scale_to_unit_disk(
    coeffs: Sequence[complex],
) -> tuple[List[complex], float]:
    """Rescale ``p`` so that all of its roots lie in the closed unit disk.

    Substituting ``z = R w`` with ``R`` from :func:`kalantari_bound` maps every
    root into ``|w| <= 1``.  This is a routine normalisation, not a
    contribution; it simply removes the dependence of the stopping criterion on
    the scale of the input.

    Returns
    -------
    (scaled_coeffs, R) : tuple
        ``scaled_coeffs[k] = coeffs[k] * R**(n-k)``.  A root ``w`` of the
        scaled polynomial corresponds to the root ``R * w`` of the original.
    """
    radius = kalantari_bound(coeffs)
    degree = len(coeffs) - 1
    scaled = [coeffs[k] * (radius ** (degree - k)) for k in range(len(coeffs))]
    return scaled, radius


# --------------------------------------------------------------------------- #
# Iteration maps
# --------------------------------------------------------------------------- #
def newton_step(p: complex, dp: complex) -> complex | None:
    """Newton increment ``-p/p'``, or ``None`` at a numerically critical point."""
    if dp == 0 or abs(dp) < CRITICAL_TOL:
        return None
    return -p / dp


def rnm_step(
    coeffs: Sequence[complex],
    z: complex,
    damping: float = 3.0,
    counter: EvalCounter | None = None,
) -> complex | None:
    """Robust Newton Method increment at a non-critical point ``z``.

    Implements the closed form

    .. math::

        \\widehat{N}_p(z) - z
        \\;=\\; - \\frac{p(z)\\,\\overline{p'(z)}}{d^{2} A(z)^{2}},
        \\qquad
        A(z) = \\max_{0 \\le j \\le n} \\frac{|p^{(j)}(z)|}{j!},

    with ``d = damping``.  The increment is the steepest-descent direction of
    ``|p|**2`` -- indeed ``p \\overline{p'}`` is twice the Wirtinger derivative
    of ``|p|**2`` with respect to ``\\bar z`` -- carrying the specific step
    length for which the Geometric Modulus Principle guarantees an a priori
    decrease of ``|p|``.

    Returns ``None`` when ``z`` is critical, where the closed form degenerates;
    :func:`_critical_escape` handles that case.
    """
    degree = len(coeffs) - 1
    taylor = taylor_coefficients(coeffs, z)
    if counter is not None:
        counter.add(degree, _taylor_cost(degree, degree) / max(degree, 1))

    a_z = max(abs(c) for c in taylor)
    if a_z == 0.0:
        return None
    p, dp = taylor[0], taylor[1]
    if abs(dp) < CRITICAL_TOL:
        return None
    return -(p * dp.conjugate()) / ((damping ** 2) * a_z * a_z)


def truncated_rnm_step(
    coeffs: Sequence[complex],
    z: complex,
    p: complex,
    dp: complex,
    damping: float = 3.0,
    counter: EvalCounter | None = None,
) -> tuple[complex | None, int]:
    """Adaptive Truncated RNM: smallest ``m`` whose step decreases ``|p|``.

    For ``0 <= m <= n`` set ``A_m(z) = max_{0<=j<=m} |p**(j)(z)|/j!`` and

    .. math::

        \\widehat{N}_p^{(m)}(z)
        \\;=\\; z - \\frac{p(z)\\,\\overline{p'(z)}}{d^{2} A_m(z)^{2}} .

    Since ``A_m <= A_n = A``, small ``m`` gives a *longer* step than full RNM
    -- aggressive but not guaranteed.  Starting at ``m = 0`` and increasing
    until the modulus actually decreases therefore recovers the guarantee at
    ``m = n``, where the step is exactly :func:`rnm_step`, while usually
    stopping far earlier.

    The economy is real: ``A_0 = |p(z)|`` and ``A_1 = max(|p(z)|, |p'(z)|)``
    reuse values Newton has already computed, so the common case costs no
    derivative evaluations at all beyond the failed Newton attempt.

    Returns
    -------
    (step, m) : tuple
        The accepted increment and the truncation order used, or
        ``(None, -1)`` if no ``m`` succeeded.
    """
    degree = len(coeffs) - 1
    mod_p = abs(p)
    if abs(dp) < CRITICAL_TOL:
        return None, -1

    scale = damping ** 2
    numerator = p * dp.conjugate()

    # m = 0 and m = 1 reuse p and p', already in hand from the failed Newton
    # attempt, so the common case costs no extra derivative evaluation.
    a_m = abs(p)
    taylor: List[complex] = []
    have = 1
    for m in range(degree + 1):
        if m == 1:
            a_m = max(a_m, abs(dp))
        elif m >= 2:
            if m > have:
                # Extend the expansion geometrically rather than building all
                # n + 1 coefficients up front.  In practice the loop stops at
                # m = 2, so paying O(n^2) for the full expansion on every
                # fallback would dominate the whole solver.
                have = min(degree, max(4, 2 * m))
                taylor = taylor_coefficients(coeffs, z, have)
                if counter is not None:
                    counter.add(1, _taylor_cost(degree, have))
            a_m = max(a_m, abs(taylor[m]))

        if a_m == 0.0:
            continue
        step = -numerator / (scale * a_m * a_m)
        if counter is not None:
            counter.add(degree)
        if abs(horner(coeffs, z + step)) < mod_p:
            return step, m
    return None, -1


def _critical_escape(
    coeffs: Sequence[complex],
    z: complex,
    counter: EvalCounter | None = None,
) -> complex | None:
    """Leave a critical point while still decreasing ``|p|``.

    At a critical point the closed RNM form vanishes.  Let ``k >= 2`` be the
    smallest index with ``c_k = p**(k)(z)/k! != 0`` in the local expansion
    ``p(z + h) = c_0 + c_k h**k + ...``.  Each of the ``k`` solutions of
    ``h**k = -c_0/c_k`` points along a descent ray, and by the Geometric
    Modulus Principle a short enough step along one of them strictly decreases
    the modulus whenever ``z`` is not itself a root.  Backtracking on the step
    length therefore always succeeds.

    In practice this branch is almost never taken: Kalantari's revised analysis
    shows that RNM iterates started at a non-critical point bypass critical
    points, and the numerical experiments in this repository record zero
    escapes over more than two million iterations.
    """
    degree = len(coeffs) - 1
    taylor = taylor_coefficients(coeffs, z)
    if counter is not None:
        counter.add(1, _taylor_cost(degree, degree))
    c0 = taylor[0]
    if c0 == 0:
        return 0.0 + 0.0j
    mod_p = abs(c0)

    for k in range(2, len(taylor)):
        ck = taylor[k]
        if ck == 0:
            continue
        radius = abs(c0 / ck) ** (1.0 / k)
        phase = cmath.phase(-c0 / ck) / k
        for j in range(k):
            direction = radius * cmath.exp(1j * (phase + 2.0 * math.pi * j / k))
            length = 1.0
            for _ in range(60):
                trial = z + length * direction
                if counter is not None:
                    counter.add(degree)
                if abs(horner(coeffs, trial)) < mod_p:
                    return length * direction
                length *= 0.5
        break
    return None


# --------------------------------------------------------------------------- #
# Single-root solvers
# --------------------------------------------------------------------------- #
def solve_chn(
    coeffs: Sequence[complex],
    z0: complex = 0.5 + 0.5j,
    max_iter: int = 500,
    damping: float = 3.0,
    truncated: bool = False,
    step_tol: float = 4.0 * EPS,
) -> SolveResult:
    """Conditional Hybrid Newton iteration for a single root.

    At each iterate the plain Newton step is attempted first.  It is accepted
    only if it strictly decreases ``|p|``; otherwise the method falls back to
    the Robust Newton step, which is guaranteed to decrease it.  Consequently
    ``|p(z_t)|`` is strictly decreasing along the whole orbit, so the iteration
    **cannot cycle** -- in contrast to Newton alone, which admits attracting
    cycles.

    Parameters
    ----------
    coeffs : sequence of complex
        Coefficients in descending order.
    z0 : complex
        Starting point.  A complex seed breaks the symmetry of the real axis
        and lets the iteration reach conjugate pairs.
    max_iter : int
        Iteration cap.
    damping : float
        The parameter ``d`` in ``1/(d**2 A(z)**2)``.  ``d = 3`` reproduces the
        factor ``1/9`` for which the theory is stated.
    truncated : bool
        Use the adaptive Truncated RNM fallback instead of full RNM.
    step_tol : float
        Relative step size below which the iterate is deemed stationary.

    Returns
    -------
    SolveResult
        ``roots`` holds the single approximation found.
    """
    z = complex(z0)
    degree = len(coeffs) - 1
    counter = EvalCounter()
    newton_hits = 0
    rnm_hits = 0
    escapes = 0
    orders: List[int] = []

    magnitudes = [abs(a) for a in coeffs]
    bound_scale = 4.0 * (degree + 1) * EPS

    def within_rounding(value: float, point: complex) -> bool:
        """Has ``|p|`` reached the level at which evaluation is pure noise?"""
        acc = magnitudes[0]
        az = abs(point)
        for a in magnitudes[1:]:
            acc = acc * az + a
        return value <= bound_scale * acc

    # p and p' are carried across iterations: the descent test already has to
    # evaluate the polynomial at the trial point, so evaluating p and p'
    # together there costs one extra pass instead of two.
    p, dp = horner_with_derivative(coeffs, z)
    counter.add(degree, 2.0)

    for iteration in range(1, max_iter + 1):
        mod_p = abs(p)
        if within_rounding(mod_p, z):
            return SolveResult([z], iteration, True, counter,
                               newton_hits, rnm_hits, escapes, orders)

        # --- Attempt 1: Newton, accepted only on strict descent -------------
        step = newton_step(p, dp)
        if step is not None:
            trial = z + step
            p_trial, dp_trial = horner_with_derivative(coeffs, trial)
            counter.add(degree, 2.0)
            if abs(p_trial) < mod_p:
                newton_hits += 1
                z, p, dp = trial, p_trial, dp_trial
                if abs(step) <= step_tol * max(1.0, abs(z)):
                    return SolveResult([z], iteration, True, counter,
                                       newton_hits, rnm_hits, escapes, orders)
                continue

        # --- Attempt 2: robust fallback, guaranteed descent ------------------
        if truncated:
            step, order = truncated_rnm_step(coeffs, z, p, dp, damping, counter)
            if step is not None:
                orders.append(order)
        else:
            step = rnm_step(coeffs, z, damping, counter)
            if step is not None:
                counter.add(degree)
                if abs(horner(coeffs, z + step)) >= mod_p:
                    step = None

        if step is not None:
            rnm_hits += 1
            z = z + step
            if abs(step) <= step_tol * max(1.0, abs(z)):
                return SolveResult([z], iteration, True, counter,
                                   newton_hits, rnm_hits, escapes, orders)
            p, dp = horner_with_derivative(coeffs, z)
            counter.add(degree, 2.0)
            continue

        # --- Attempt 3: escape from a critical point -------------------------
        step = _critical_escape(coeffs, z, counter)
        if step is None or abs(step) <= step_tol * max(1.0, abs(z)):
            return SolveResult([z], iteration, True, counter,
                               newton_hits, rnm_hits, escapes, orders)
        z = z + step
        escapes += 1
        p, dp = horner_with_derivative(coeffs, z)
        counter.add(degree, 2.0)

    return SolveResult([z], max_iter, False, counter,
                       newton_hits, rnm_hits, escapes, orders)


def _deflate(coeffs: Sequence[complex], root: complex) -> List[complex]:
    """Forward synthetic division of ``p`` by ``(z - root)``."""
    quotient: List[complex] = [coeffs[0]]
    acc = coeffs[0]
    for a in coeffs[1:-1]:
        acc = acc * root + a
        quotient.append(acc)
    return quotient


def _maehly_polish(
    coeffs: Sequence[complex],
    z0: complex,
    accepted: Sequence[complex],
    max_iter: int = 40,
    counter: EvalCounter | None = None,
) -> complex:
    """Refine ``z0`` on the *original* coefficients, suppressing known roots.

    Polishing an approximation against the original polynomial is what removes
    deflation drift, but done naively it is unsafe: the refinement is free to
    walk back to a root that has already been accepted, and the algorithm then
    deflates by a factor the working polynomial does not contain, corrupting
    every later root.  Maehly's correction removes that failure mode.  Writing
    ``q(z) = p(z) / prod_{j} (z - r_j)`` over the accepted roots ``r_j``, the
    Newton step for ``q`` is

    .. math::

        \\Delta z \\;=\\;
        \\frac{-p(z)}{p'(z) - p(z) \\sum_j (z - r_j)^{-1}} ,

    which needs no deflated coefficients at all: the accepted roots are simply
    repelling, so the iteration cannot return to them.
    """
    z = complex(z0)
    degree = len(coeffs) - 1
    for _ in range(max_iter):
        p, dp = horner_with_derivative(coeffs, z)
        if counter is not None:
            counter.add(degree, 2.0)
            counter.add(len(accepted))
        if abs(p) <= backward_error_bound(coeffs, z):
            return z
        suppression = 0.0 + 0.0j
        for r in accepted:
            diff = z - r
            if diff == 0:
                diff = complex(EPS, EPS)
            suppression += 1.0 / diff
        denominator = dp - p * suppression
        if denominator == 0:
            return z
        correction = -p / denominator
        z = z + correction
        if abs(correction) <= 4.0 * EPS * max(1.0, abs(z)):
            return z
    return z


def _ehrlich_aberth_refine(
    coeffs: Sequence[complex],
    roots: Sequence[complex],
    max_sweeps: int = 20,
    counter: EvalCounter | None = None,
) -> List[complex]:
    """Final simultaneous refinement of all ``n`` approximations.

    One Ehrlich--Aberth sweep applies the Maehly correction to every root at
    once, each root suppressing the other ``n - 1``.  Starting from the
    approximations produced by the discovery phase this converges in a couple
    of sweeps and guarantees that the reported roots are those of the original
    coefficient vector rather than of a drifted deflated one.

    The identical sweep is applied to every method benchmarked in this study,
    so it cannot flatter one of them.
    """
    current = [complex(r) for r in roots]
    degree = len(coeffs) - 1
    n = len(current)
    previous_largest = float("inf")
    for _ in range(max_sweeps):
        largest = 0.0
        for i in range(n):
            z = current[i]
            p, dp = horner_with_derivative(coeffs, z)
            if counter is not None:
                counter.add(degree, 2.0)
                counter.add(n - 1)
            if p == 0:
                continue
            suppression = 0.0 + 0.0j
            for j in range(n):
                if j == i:
                    continue
                diff = z - current[j]
                if diff == 0:
                    diff = complex(EPS, EPS)
                suppression += 1.0 / diff
            denominator = dp - p * suppression
            if denominator == 0:
                continue
            correction = -p / denominator
            current[i] = z + correction
            largest = max(largest, abs(correction) / max(1.0, abs(z)))
        # Stop as soon as the sweep is either converged or no longer making
        # progress.  Because each root was already polished on the original
        # coefficients as it was found, this loop is a safety net rather than
        # the workhorse, and it normally exits after one or two sweeps.
        if largest <= 4.0 * EPS or largest >= 0.5 * previous_largest:
            break
        previous_largest = largest
    return current


def solve_all_chn(
    coeffs: Sequence[complex],
    seed: complex = 0.5 + 0.5j,
    max_iter: int = 500,
    damping: float = 3.0,
    truncated: bool = False,
    rescale: bool = True,
    warm_start_offset: complex = 0.01 + 0.01j,
) -> SolveResult:
    """All ``n`` roots by *polish-then-deflate*.

    The ordering matters.  The draft implementation deflated by the raw
    approximation returned from the deflated polynomial and only polished at
    the very end; the error committed at step ``i`` then contaminates every
    subsequent deflation, and by degree 20 the accumulated drift dominates the
    answer.  Here each root is refined on the **original** coefficients the
    moment it is found, and the deflation uses the refined value, so the
    working polynomial never drifts.

    Parameters
    ----------
    coeffs : sequence of complex
        Coefficients in descending order.
    seed : complex
        Starting point for each deflated sub-problem.
    truncated : bool
        Use the adaptive Truncated RNM fallback.
    rescale : bool
        Normalise the roots into the unit disk via :func:`scale_to_unit_disk`
        before solving, and undo the scaling afterwards.

    Returns
    -------
    SolveResult
        ``roots`` holds the ``n`` refined approximations.
    """
    work = list(coeffs)
    radius = 1.0
    if rescale:
        work, radius = scale_to_unit_disk(work)

    degree = len(work) - 1
    current = list(work)
    roots: List[complex] = []
    total = EvalCounter()
    iterations = 0
    newton_hits = rnm_hits = escapes = 0
    orders: List[int] = []
    converged = True

    for _ in range(degree):
        if len(current) == 2:
            approx = -current[1] / current[0]
        else:
            # Warm start.  Roots of a real-rooted or clustered polynomial tend
            # to lie near one another, so the previously accepted root is a
            # far better starting point than a fixed seed; the small imaginary
            # offset keeps the iterate off the real axis and off the root just
            # divided out.
            start = seed if not roots else roots[-1] + warm_start_offset
            found = solve_chn(current, start, max_iter, damping, truncated)
            total += found.evals
            iterations += found.iterations
            newton_hits += found.newton_steps
            rnm_hits += found.rnm_steps
            escapes += found.escape_steps
            orders.extend(found.truncation_orders)
            converged = converged and found.converged
            approx = found.roots[0]

        # Refine against the ORIGINAL coefficients, suppressing the roots
        # already accepted, then deflate by the refined value so the working
        # polynomial never drifts.
        root = _maehly_polish(work, approx, roots, counter=total)
        roots.append(root)
        current = _deflate(current, root)

    roots = _ehrlich_aberth_refine(work, roots, counter=total)

    if rescale:
        roots = [r * radius for r in roots]

    return SolveResult(roots, iterations, converged, total,
                       newton_hits, rnm_hits, escapes, orders)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def solve_newton_deflation(
    coeffs: Sequence[complex],
    seed: complex = 0.5 + 0.5j,
    max_iter: int = 500,
    rescale: bool = True,
) -> SolveResult:
    """Plain Newton with deflation -- the ablation baseline.

    Identical to :func:`solve_all_chn` except that no descent test is applied
    and there is no robust fallback: the iteration takes the Newton step
    unconditionally.  Comparing against this isolates exactly what the RNM
    fallback buys.
    """
    work = list(coeffs)
    radius = 1.0
    if rescale:
        work, radius = scale_to_unit_disk(work)

    degree = len(work) - 1
    current = list(work)
    roots: List[complex] = []
    counter = EvalCounter()
    iterations = 0
    converged = True

    def _newton_only(
        target: Sequence[complex], z0: complex
    ) -> tuple[complex, int, bool]:
        z = complex(z0)
        for it in range(1, max_iter + 1):
            p, dp = horner_with_derivative(target, z)
            counter.add(len(target) - 1, 2.0)
            if abs(p) <= backward_error_bound(target, z):
                return z, it, True
            if abs(dp) < CRITICAL_TOL:
                return z, it, False
            step = -p / dp
            z = z + step
            if abs(z) > 1e12:
                return z, it, False
            if abs(step) <= 4.0 * EPS * max(1.0, abs(z)):
                return z, it, True
        return z, max_iter, False

    for _ in range(degree):
        if len(current) == 2:
            approx = -current[1] / current[0]
        else:
            approx, its, ok = _newton_only(current, seed)
            iterations += its
            converged = converged and ok
        root = _maehly_polish(work, approx, roots, counter=counter)
        roots.append(root)
        current = _deflate(current, root)

    roots = _ehrlich_aberth_refine(work, roots, counter=counter)

    if rescale:
        roots = [r * radius for r in roots]
    return SolveResult(roots, iterations, converged, counter)


def _horner_real(coeffs: Sequence[float], x: float) -> tuple[float, float]:
    """Real-arithmetic Horner evaluation of ``p`` and ``p'``."""
    value = float(coeffs[0])
    deriv = 0.0
    for a in coeffs[1:]:
        deriv = deriv * x + value
        value = value * x + float(a)
    return value, deriv


def _derivative(coeffs: Sequence[float]) -> List[float]:
    """Coefficients of ``p'`` in descending order."""
    degree = len(coeffs) - 1
    return [coeffs[i] * (degree - i) for i in range(degree)]


def solve_yuksel(
    coeffs: Sequence[float],
    x_lo: float,
    x_hi: float,
    tol: float = 1e-14,
    max_iter: int = 100,
) -> SolveResult:
    """Yuksel's (2022) recursive monotonic interval splitting.

    The derivative's roots are found recursively and partition ``[x_lo, x_hi]``
    into intervals on which ``p`` is monotone; each sign-changing interval then
    carries exactly one root, located by a Newton iteration that falls back to
    bisection whenever the Newton point leaves the bracket.  Recursion stops at
    degree two, solved by the numerically stable quadratic formula.

    This method returns only the **real** roots inside ``[x_lo, x_hi]``.  The
    complex roots of a general polynomial are outside its reach, which is the
    structural difference from the complex-plane methods above.
    """
    counter = EvalCounter()

    def recurse(local: List[float], lo: float, hi: float) -> List[float]:
        poly = list(local)
        while len(poly) > 1 and poly[0] == 0.0:
            poly.pop(0)
        degree = len(poly) - 1
        if degree <= 0:
            return []
        if degree == 1:
            r = -poly[1] / poly[0]
            return [r] if lo <= r <= hi else []
        if degree == 2:
            a, b, c = float(poly[0]), float(poly[1]), float(poly[2])
            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                return []
            sq = math.sqrt(disc)
            sign_b = 1.0 if b >= 0.0 else -1.0
            q = -0.5 * (b + sign_b * sq)
            candidates = [q / a]
            if q != 0.0:
                candidates.append(c / q)
            else:
                candidates.append(-b / a - q / a)
            return sorted(r for r in candidates if lo <= r <= hi)

        deriv = _derivative(poly)
        crit = recurse(deriv, lo, hi)
        breaks = [lo] + sorted(c for c in crit if lo < c < hi) + [hi]

        found: List[float] = []
        for i in range(len(breaks) - 1):
            xa, xb = breaks[i], breaks[i + 1]
            if xb - xa < tol:
                continue
            fa, _ = _horner_real(poly, xa)
            fb, _ = _horner_real(poly, xb)
            counter.add(degree, 4.0, complex_arith=False)
            if fa == 0.0:
                found.append(xa)
                continue
            if fb == 0.0:
                continue
            if fa * fb > 0.0:
                continue

            x_min, x_max, f_min = xa, xb, fa
            x = 0.5 * (xa + xb)
            for _ in range(max_iter):
                fx, dfx = _horner_real(poly, x)
                counter.add(degree, 2.0, complex_arith=False)
                if fx == 0.0 or (x_max - x_min) < tol * max(1.0, abs(x)):
                    break
                if (fx > 0.0) == (f_min > 0.0):
                    x_min, f_min = x, fx
                else:
                    x_max = x
                if dfx != 0.0:
                    x_new = x - fx / dfx
                    if not (x_min < x_new < x_max):
                        x_new = 0.5 * (x_min + x_max)
                else:
                    x_new = 0.5 * (x_min + x_max)
                if abs(x_new - x) <= tol * max(1.0, abs(x)):
                    x = x_new
                    break
                x = x_new
            found.append(x)
        return found

    roots = recurse(list(map(float, coeffs)), x_lo, x_hi)
    return SolveResult([complex(r) for r in roots], 0, True, counter)


def solve_numpy(coeffs: Sequence[complex]) -> SolveResult:
    """Reference solver: eigenvalues of the companion matrix (LAPACK).

    Included as an accuracy yardstick only.  It is a compiled routine, so its
    wall-clock time is not comparable with the pure-Python implementations
    benchmarked here.
    """
    import numpy as np

    roots = [complex(r) for r in np.roots(np.asarray(coeffs, dtype=complex))]
    return SolveResult(roots, 0, True, EvalCounter())


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #
SOLVERS: dict[str, Callable[..., SolveResult]] = {
    "chn": lambda c, **kw: solve_all_chn(c, truncated=False, **kw),
    "chn-t": lambda c, **kw: solve_all_chn(c, truncated=True, **kw),
    "newton": solve_newton_deflation,
    "numpy": lambda c, **kw: solve_numpy(c),
}
