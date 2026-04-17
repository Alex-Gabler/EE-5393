"""
EE 5393 Homework #3 – Stochastic Logic Synthesis
Alex, April 2026

Problems:
  1. Synthesizing Stochastic Logic (Bernstein polynomial circuits)
     (a) f(x) = x - x^2/4
     (b) Approximate cos(x)
     (c) Implement 31t^5/32 + 5t^4/32 - 5t^3/8 + 5t^2/4 - 5t/4 + 1/2
  2. Transforming Probabilities
     (a) Generate decimal probabilities from {0.4, 0.5} using AND + NOT
     (b) Generate binary-fraction probabilities from {0.5} using AND + NOT

Usage:
  python hw3.py        # run everything
  python hw3.py 1      # Problem 1 only
  python hw3.py 2      # Problem 2 only
"""

import sys
import numpy as np
from fractions import Fraction
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# PROBLEM 1: BERNSTEIN POLYNOMIAL CIRCUITS
# ============================================================

def poly_to_bernstein(coeffs, degree=None):
    """
    Convert standard polynomial coefficients to Bernstein coefficients.

    coeffs[i] = coefficient of t^i  (starting from constant term).
    degree = Bernstein degree (default: len(coeffs)-1).

    Conversion formula (Lorentz 1953):
        b_k = sum_{j=0}^{k} C(k,j) / C(n,j) * a_j

    Returns list of Bernstein coefficients b_0, ..., b_n.
    """
    from math import comb
    n = degree if degree is not None else len(coeffs) - 1
    # Pad coefficients with zeros up to degree n
    a = list(coeffs) + [Fraction(0)] * (n + 1 - len(coeffs))
    a = [Fraction(x) for x in a]
    b = []
    for k in range(n + 1):
        bk = Fraction(0)
        for j in range(k + 1):
            bk += Fraction(comb(k, j), comb(n, j)) * a[j]
        b.append(bk)
    return b


def eval_bernstein(b, t):
    """Evaluate Bernstein polynomial at t using the Bernstein basis."""
    from math import comb
    n = len(b) - 1
    result = Fraction(0)
    t = Fraction(t)
    for k in range(n + 1):
        basis = Fraction(comb(n, k)) * t**k * (1 - t)**(n - k)
        result += Fraction(b[k]) * basis
    return result


def eval_poly(coeffs, t):
    """Evaluate standard polynomial at t (coeffs[i] = coeff of t^i)."""
    result = Fraction(0)
    t = Fraction(t)
    for i, a in enumerate(coeffs):
        result += Fraction(a) * t**i
    return result


def run_problem1():
    print("=" * 65)
    print("PROBLEM 1: BERNSTEIN POLYNOMIAL STOCHASTIC CIRCUITS")
    print("=" * 65)

    # ─── (a) f(x) = x - x^2/4 ───────────────────────────────────────
    print("\n─── (a)  f(x) = x - x²/4 ───")
    # Coefficients: a0=0, a1=1, a2=-1/4
    coeffs_a = [Fraction(0), Fraction(1), Fraction(-1, 4)]
    b_a = poly_to_bernstein(coeffs_a)
    print(f"  Degree: {len(b_a) - 1}")
    print(f"  Bernstein coefficients: {[str(x) for x in b_a]}")
    print(f"  Decimal:                {[float(x) for x in b_a]}")
    print(f"  All in [0,1]: {all(0 <= b <= 1 for b in b_a)}")
    # Verify at a few points
    print(f"  Verification (direct vs Bernstein):")
    for t in [Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1)]:
        direct = eval_poly(coeffs_a, t)
        bern = eval_bernstein(b_a, t)
        match = direct == bern
        print(f"    t={float(t):.2f}: f(t)={float(direct):.6f}  Bernstein={float(bern):.6f}  match={match}")
    print()
    print("  Circuit description:")
    print("    - Two independent stochastic input streams, both with probability x")
    print("    - '+' block counts the number of 1s (0, 1, or 2)")
    print("    - MUX selects:")
    print(f"        count=0 → constant stream with prob b₀ = {b_a[0]}  = {float(b_a[0])}")
    print(f"        count=1 → constant stream with prob b₁ = {b_a[1]} = {float(b_a[1])}")
    print(f"        count=2 → constant stream with prob b₂ = {b_a[2]} = {float(b_a[2])}")

    # ─── (b) cos(x) approximation ────────────────────────────────────
    print("\n─── (b)  cos(x) approximation ───")
    # Taylor series: cos(x) ≈ 1 - x^2/2 + x^4/24 (degree 4)
    # a0=1, a1=0, a2=-1/2, a3=0, a4=1/24
    coeffs_b = [Fraction(1), Fraction(0), Fraction(-1, 2), Fraction(0), Fraction(1, 24)]
    b_b = poly_to_bernstein(coeffs_b)
    print(f"  Approximation: cos(x) ≈ 1 - x²/2 + x⁴/24  (degree-4 Taylor truncation)")
    print(f"  Bernstein coefficients: {[str(x) for x in b_b]}")
    print(f"  Decimal:                {[float(x) for x in b_b]}")
    print(f"  All in [0,1]: {all(0 <= b <= 1 for b in b_b)}")
    print(f"  Range of approximation on [0,1]: [{float(min(eval_poly(coeffs_b, t) for t in [Fraction(k,100) for k in range(101)])):.4f}, "
          f"{float(max(eval_poly(coeffs_b, t) for t in [Fraction(k,100) for k in range(101)])):.4f}]")
    print()
    print("  Verification (polynomial vs exact cos):")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        poly_val = float(eval_poly(coeffs_b, Fraction(t).limit_denominator(1000)))
        cos_val = float(np.cos(t))
        print(f"    t={t:.2f}: poly={poly_val:.6f}  cos(t)={cos_val:.6f}  error={abs(poly_val-cos_val):.2e}")
    print()
    print("  Circuit description:")
    print("    - Four independent stochastic input streams, all with probability x")
    print("    - '+' block counts ones (0, 1, 2, 3, or 4)")
    print("    - MUX selects:")
    for k, bk in enumerate(b_b):
        print(f"        count={k} → prob b_{k} = {bk} ≈ {float(bk):.6f}")

    # ─── (c) degree-5 polynomial ─────────────────────────────────────
    print("\n─── (c)  p(t) = 31t⁵/32 + 5t⁴/32 - 5t³/8 + 5t²/4 - 5t/4 + 1/2 ───")
    # a0=1/2, a1=-5/4, a2=5/4, a3=-5/8, a4=5/32, a5=31/32
    coeffs_c = [
        Fraction(1, 2),
        Fraction(-5, 4),
        Fraction(5, 4),
        Fraction(-5, 8),
        Fraction(5, 32),
        Fraction(31, 32),
    ]
    b_c = poly_to_bernstein(coeffs_c)
    print(f"  Bernstein coefficients (degree {len(b_c)-1}):")
    for k, bk in enumerate(b_c):
        print(f"    b_{k} = {bk} = {float(bk):.8f}")
    print(f"  All in [0,1]: {all(0 <= b <= 1 for b in b_c)}")
    print()
    print("  Circuit description:")
    print("    - Five independent stochastic input streams, all with probability t")
    print("    - '+' block counts ones (0 to 5)")
    print("    - MUX selects based on count:")
    for k, bk in enumerate(b_c):
        print(f"        count={k} → prob b_{k} = {bk} ≈ {float(bk):.8f}")
    print()
    print("  Test values:")
    test_vals = [Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1)]
    print(f"  {'X':>6}  {'p(X) exact':>16}  {'p(X) decimal':>14}  {'Bernstein':>14}")
    print(f"  {'─'*6}  {'─'*16}  {'─'*14}  {'─'*14}")
    for t in test_vals:
        direct = eval_poly(coeffs_c, t)
        bern = eval_bernstein(b_c, t)
        match = direct == bern
        print(f"  {float(t):>6.2f}  {str(direct):>16}  {float(direct):>14.8f}  {float(bern):>14.8f}  {'✓' if match else '✗'}")


# ============================================================
# PROBLEM 2: PROBABILITY SYNTHESIS
# ============================================================

def synthesize_prob_bfs(target, sources, max_depth=35):
    """
    BFS to find shortest circuit from `sources` to generate `target`
    using AND and NOT gates.

    Works BACKWARD from target: each operation undoes one gate to
    find what the input to that gate must have been.
      - NOT:     z → 1-z
      - AND(p):  z → z/p  (for each p in sources)

    Returns (ops_forward, final_input) where ops_forward is the
    sequence of gates from input to output.
    """
    target = Fraction(target).limit_denominator(10**12)
    sources = [Fraction(s) for s in sources]

    # BFS state: (current_z, backward_ops_list)
    queue = deque([(target, [])])
    visited = {target}

    while queue:
        z, ops = queue.popleft()

        # Base cases
        if z in sources or z == Fraction(0) or z == Fraction(1):
            # Reconstruct forward circuit
            forward = list(reversed(ops))
            return forward, z

        if len(ops) >= max_depth:
            continue

        # Try NOT
        new_z = Fraction(1) - z
        if Fraction(0) <= new_z <= Fraction(1) and new_z not in visited:
            visited.add(new_z)
            queue.append((new_z, ops + [('NOT', z, new_z)]))

        # Try AND with each source probability
        for p in sources:
            new_z = z / p
            if Fraction(0) < new_z <= Fraction(1) and new_z not in visited:
                visited.add(new_z)
                queue.append((new_z, ops + [(f'AND({float(p):.1f})', z, new_z)]))

    return None, None


def run_problem2():
    print("\n" + "=" * 65)
    print("PROBLEM 2: TRANSFORMING PROBABILITIES")
    print("=" * 65)

    # ─── (a) from {0.4, 0.5} ─────────────────────────────────────────
    print("\n─── (a) Target probabilities from S = {0.4, 0.5} ───")
    sources_a = [Fraction(2, 5), Fraction(1, 2)]

    targets_a = [
        ("0.8881188",  Fraction(8881188,  10000000)),
        ("0.2119209",  Fraction(2119209,  10000000)),
        ("0.5555555",  Fraction(5555555,  10000000)),
    ]

    for name, target in targets_a:
        print(f"\n  Target: {name} = {target} ≈ {float(target):.7f}")
        ops, base = synthesize_prob_bfs(target, sources_a, max_depth=40)
        if ops is None:
            print("    NOT FOUND within depth limit")
            continue
        print(f"  Base input: {base} = {float(base):.4f}  ({len(ops)} gates total)")
        print(f"  Forward circuit (input → output):")
        # Reconstruct intermediate probabilities
        z = base
        print(f"    Start:  p = {z} = {float(z):.8f}")
        for op_name, out_prob, in_prob in ops:
            if op_name == 'NOT':
                new_z = Fraction(1) - z
                print(f"    NOT:    p = {new_z} = {float(new_z):.8f}")
                z = new_z
            elif 'AND' in op_name:
                # Extract the probability from the op name
                p_str = op_name.split('(')[1].rstrip(')')
                p = Fraction(p_str).limit_denominator(100)
                new_z = z * p
                print(f"    {op_name}: p = {new_z} = {float(new_z):.8f}")
                z = new_z
        print(f"    Final:  {float(z):.8f}  (target: {float(target):.8f})  "
              f"{'✓' if abs(float(z) - float(target)) < 1e-9 else '✗'}")

    # ─── (b) from {0.5} only, binary fractions ───────────────────────
    print("\n─── (b) Binary-fraction targets from S = {0.5} ───")
    sources_b = [Fraction(1, 2)]

    targets_b = [
        ("0.1011111₂", Fraction(95,  128)),
        ("0.1101111₂", Fraction(111, 128)),
        ("0.1010111₂", Fraction(87,  128)),
    ]

    for name, target in targets_b:
        print(f"\n  Target: {name} = {target} = {float(target):.7f}")
        ops, base = synthesize_prob_bfs(target, sources_b, max_depth=20)
        if ops is None:
            print("    NOT FOUND within depth limit")
            continue
        print(f"  Base input: {base}  ({len(ops)} gates total)")
        print(f"  Forward circuit (input → output):")
        z = base
        print(f"    Start:  p = {z} = {float(z):.8f}")
        for op_name, out_prob, in_prob in ops:
            if op_name == 'NOT':
                new_z = Fraction(1) - z
                print(f"    NOT:    p = {new_z} = {float(new_z):.8f}")
                z = new_z
            elif 'AND' in op_name:
                p = Fraction(1, 2)
                new_z = z * p
                print(f"    AND(0.5): p = {new_z} = {float(new_z):.8f}")
                z = new_z
        print(f"    Final:  {float(z):.8f}  (target: {float(target):.8f})  "
              f"{'✓' if abs(float(z) - float(target)) < 1e-9 else '✗'}")


# ============================================================
# PLOTTING
# ============================================================

def make_plots():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    t = np.linspace(0, 1, 200)

    # Plot 1(a)
    ax = axes[0]
    y = t - t**2 / 4
    ax.plot(t, y, 'b-', linewidth=2, label=r'$f(x) = x - x^2/4$')
    ax.plot(t, t, 'k--', alpha=0.3, label='y = x (diagonal)')
    ax.set_title('Problem 1(a): f(x) = x − x²/4')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Plot 1(b)
    ax = axes[1]
    y_cos = np.cos(t)
    y_approx = 1 - t**2/2 + t**4/24
    ax.plot(t, y_cos, 'g-', linewidth=2, label='cos(x)')
    ax.plot(t, y_approx, 'r--', linewidth=2, label='1 − x²/2 + x⁴/24')
    ax.set_title('Problem 1(b): cos(x) approximation')
    ax.set_xlabel('x')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Plot 1(c)
    ax = axes[2]
    y_c = (31*t**5/32 + 5*t**4/32 - 5*t**3/8 + 5*t**2/4 - 5*t/4 + 0.5)
    ax.plot(t, y_c, 'm-', linewidth=2, label='p(t)')
    test_x = [0, 0.25, 0.5, 0.75, 1.0]
    test_y_exact = []
    for tx in test_x:
        coeffs_c = [Fraction(1,2), Fraction(-5,4), Fraction(5,4),
                    Fraction(-5,8), Fraction(5,32), Fraction(31,32)]
        val = eval_poly(coeffs_c, Fraction(tx).limit_denominator(1000))
        test_y_exact.append(float(val))
    ax.scatter(test_x, test_y_exact, color='red', zorder=5, s=80, label='Test points')
    ax.set_title('Problem 1(c): degree-5 polynomial')
    ax.set_xlabel('t')
    ax.set_ylabel('p(t)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig('hw3_plots.png', dpi=150)
    print("\nPlots saved to hw3_plots.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if mode in ('1', 'all'):
        run_problem1()

    if mode in ('2', 'all'):
        run_problem2()

    if mode == 'all':
        make_plots()
