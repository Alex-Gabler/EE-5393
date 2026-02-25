"""
EE 5393 Homework #1 - Combined Submission
Alex, February 2026

Usage:
  python hw1.py          <- interactive menu
  python hw1.py 1a       <- run Problem 1(a) directly
  python hw1.py 1b       <- run Problem 1(b) directly
  python hw1.py 2        <- run Problem 2 directly
  python hw1.py 3a       <- run Problem 3(a) directly
  python hw1.py 3b       <- run Problem 3(b) directly
"""

import sys
import numpy as np
import math
import time
from math import comb
from collections import defaultdict


# ===========================================================================
# PROBLEM 1(a) and 1(b) - Shared CRN definition
#
# Reactions:
#   R1: 2X1 + X2 -> 4X3,  k1 = 1
#   R2: X1 + 2X3 -> 3X2,  k2 = 2
#   R3: X2 + X3  -> 2X1,  k3 = 3
#
# Discrete propensities (combinatorial):
#   a1 = k1 * C(x1,2) * x2 = x1*(x1-1)/2 * x2
#   a2 = k2 * x1 * C(x3,2) = 2 * x1 * x3*(x3-1)/2 = x1*x3*(x3-1)
#   a3 = k3 * x2 * x3       = 3 * x2 * x3
# ===========================================================================

# State change vectors: R1=[-2,-1,+4], R2=[-1,+3,-2], R3=[+2,-1,-1]
_P1_STOICH = np.array([[-2, -1, +4], [-1, +3, -2], [+2, -1, -1]])


def _p1_propensities(x1, x2, x3):
    a1 = 0.5 * x1 * (x1 - 1) * x2
    a2 = x1 * x3 * (x3 - 1)
    a3 = 3.0 * x2 * x3
    return a1, a2, a3


def _p1_step(state):
    """One Gillespie step; returns (new_state, tau)."""
    x1, x2, x3 = state
    a1, a2, a3 = _p1_propensities(x1, x2, x3)
    a_total = a1 + a2 + a3
    if a_total == 0:
        return state, float('inf')
    tau = np.random.exponential(1.0 / a_total)
    r = np.random.random() * a_total
    reaction = 0 if r < a1 else (1 if r < a1 + a2 else 2)
    return state + _P1_STOICH[reaction], tau


# ---------------------------------------------------------------------------
# Problem 1(a)
# ---------------------------------------------------------------------------

def _p1a_check_outcome(state):
    x1, x2, x3 = state
    outcomes = []
    if x1 >= 150: outcomes.append('C1')
    if x2 < 10:   outcomes.append('C2')
    if x3 > 100:  outcomes.append('C3')
    return outcomes


def _p1a_run_trial(initial_state, max_steps=1000000):
    state = np.array(initial_state, dtype=int)
    for _ in range(max_steps):
        outcomes = _p1a_check_outcome(state)
        if outcomes:
            return outcomes
        state, tau = _p1_step(state)
        if np.any(state < 0):
            return ['INVALID']
    return ['TIMEOUT']


def run_problem_1a():
    initial_state = [110, 26, 55]
    num_trials = 100000

    print("EE 5393 HW1 - Problem 1(a)")
    print(f"Running {num_trials} Gillespie SSA trials from S = {initial_state}")
    print("Outcomes: C1 (x1>=150), C2 (x2<10), C3 (x3>100)")
    print()

    np.random.seed(42)
    counts = {'C1': 0, 'C2': 0, 'C3': 0, 'TIMEOUT': 0, 'INVALID': 0}

    for trial in range(num_trials):
        for o in _p1a_run_trial(initial_state):
            counts[o] += 1
        if (trial + 1) % 10000 == 0:
            print(f"  Completed {trial + 1}/{num_trials} trials...")

    print()
    print("Results:")
    print(f"  Pr(C1) = {counts['C1']}/{num_trials} = {counts['C1']/num_trials:.6f}")
    print(f"  Pr(C2) = {counts['C2']}/{num_trials} = {counts['C2']/num_trials:.6f}")
    print(f"  Pr(C3) = {counts['C3']}/{num_trials} = {counts['C3']/num_trials:.6f}")
    if counts['TIMEOUT']:
        print(f"  (Timeouts: {counts['TIMEOUT']})")


# ---------------------------------------------------------------------------
# Problem 1(b)
# ---------------------------------------------------------------------------

def _p1b_run_n_steps(initial_state, n_steps):
    state = np.array(initial_state, dtype=int)
    for _ in range(n_steps):
        x1, x2, x3 = state
        a1, a2, a3 = _p1_propensities(x1, x2, x3)
        a_total = a1 + a2 + a3
        if a_total == 0:
            break
        r = np.random.random() * a_total
        reaction = 0 if r < a1 else (1 if r < a1 + a2 else 2)
        state = state + _P1_STOICH[reaction]
    return state


def run_problem_1b():
    initial_state = [9, 8, 7]
    n_steps = 7
    num_trials = 1000000

    print("EE 5393 HW1 - Problem 1(b)")
    print(f"Running {num_trials} trials of {n_steps} steps from S = {initial_state}")
    print()

    np.random.seed(42)
    x1_vals = np.zeros(num_trials)
    x2_vals = np.zeros(num_trials)
    x3_vals = np.zeros(num_trials)

    for trial in range(num_trials):
        fs = _p1b_run_n_steps(initial_state, n_steps)
        x1_vals[trial], x2_vals[trial], x3_vals[trial] = fs
        if (trial + 1) % 200000 == 0:
            print(f"  Completed {trial + 1}/{num_trials} trials...")

    print()
    print("Results after 7 steps:")
    print(f"  X1: mean = {np.mean(x1_vals):.4f}, variance = {np.var(x1_vals):.4f}")
    print(f"  X2: mean = {np.mean(x2_vals):.4f}, variance = {np.var(x2_vals):.4f}")
    print(f"  X3: mean = {np.mean(x3_vals):.4f}, variance = {np.var(x3_vals):.4f}")
    print()
    print("  Sanity check - net molecule conservation:")
    print(f"    Initial total: {sum(initial_state)}")
    print(f"    Mean final total: {np.mean(x1_vals + x2_vals + x3_vals):.4f}")
    print(f"    (R1 adds +1 net molecule per firing; R2,R3 are net zero)")


# ===========================================================================
# PROBLEM 2 - Lambda phage stealth vs hijack
# ===========================================================================

def _p2_parse_reactions(filename):
    reactions = []
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(':')
            if len(parts) != 3:
                print(f"Warning: skipping malformed line {line_num}: {line}")
                continue
            reactants = _p2_parse_species(parts[0].strip())
            products  = _p2_parse_species(parts[1].strip())
            rate = float(parts[2].strip())
            reactions.append({'reactants': reactants, 'products': products, 'rate': rate})
    return reactions


def _p2_parse_species(s):
    result = {}
    if not s.strip():
        return result
    tokens = s.split()
    i = 0
    while i < len(tokens):
        sp = tokens[i]
        n  = int(tokens[i + 1])
        result[sp] = result.get(sp, 0) + n
        i += 2
    return result


def _p2_parse_initial(filename):
    initial_state = {}
    stopping_conditions = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            species    = tokens[0]
            count      = int(tokens[1])
            cond_type  = tokens[2]
            initial_state[species] = count
            if cond_type == 'GE':
                stopping_conditions.append((species, int(tokens[3])))
    return initial_state, stopping_conditions


def _p2_propensity(reaction, state):
    prop = reaction['rate']
    for sp, n in reaction['reactants'].items():
        x = state.get(sp, 0)
        if x < n:
            return 0.0
        prop *= comb(x, n)
    return prop


def _p2_run_trial(reactions, initial_state, stopping_conditions,
                  max_time=5000.0, max_steps=500000):
    state = dict(initial_state)
    t = 0.0
    for _ in range(max_steps):
        for sp, threshold in stopping_conditions:
            if state.get(sp, 0) >= threshold:
                return sp
        props = [_p2_propensity(r, state) for r in reactions]
        a_total = sum(props)
        if a_total == 0:
            return 'STUCK'
        tau = np.random.exponential(1.0 / a_total)
        t += tau
        if t > max_time:
            return 'TIMEOUT'
        r = np.random.random() * a_total
        cumsum = 0.0
        chosen = len(reactions) - 1
        for i, a in enumerate(props):
            cumsum += a
            if r < cumsum:
                chosen = i
                break
        rxn = reactions[chosen]
        for sp, n in rxn['reactants'].items():
            state[sp] = state.get(sp, 0) - n
        for sp, n in rxn['products'].items():
            state[sp] = state.get(sp, 0) + n
    return 'MAX_STEPS'


def run_problem_2():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    r_file  = os.path.join(script_dir, 'lambda.r')
    in_file = os.path.join(script_dir, 'lambda.in')

    reactions, (initial_state, stopping_conditions) = (
        _p2_parse_reactions(r_file),
        _p2_parse_initial(in_file)
    )

    print("EE 5393 HW1 - Problem 2")
    print(f"Loaded {len(reactions)} reactions, {len(initial_state)} species")
    print(f"Stopping conditions (from lambda.in): {stopping_conditions}")
    # Note: lambda.in uses GE (>=); the PDF text says > 145 / > 55.
    # We use the professor's own .in file definition (>=).
    print()

    num_trials = 50
    print(f"Simulating MOI = 1..10, {num_trials} trials each")
    print(f"Stealth: cI2 >= 145  |  Hijack: Cro2 >= 55")
    print()

    results = {}
    for moi in range(1, 11):
        stealth = hijack = other = 0
        t_start = time.time()
        for trial in range(num_trials):
            trial_state = dict(initial_state)
            trial_state['MOI'] = moi
            outcome = _p2_run_trial(reactions, trial_state, stopping_conditions)
            if outcome == 'cI2':
                stealth += 1
            elif outcome == 'Cro2':
                hijack += 1
            else:
                other += 1
            if (trial + 1) % 25 == 0:
                sys.stdout.write(f"  MOI={moi}: {trial+1}/{num_trials} done\n")
                sys.stdout.flush()
        p_s = stealth / num_trials
        p_h = hijack / num_trials
        results[moi] = (p_s, p_h, other)
        print(f"MOI={moi:2d}: Pr(stealth)={p_s:.4f}, Pr(hijack)={p_h:.4f}, "
              f"other={other}/{num_trials}  [{time.time()-t_start:.1f}s]")

    print()
    print(f"{'MOI':>4s} {'Pr(stealth)':>12s} {'Pr(hijack)':>12s}")
    print("-" * 32)
    for moi in range(1, 11):
        p_s, p_h, _ = results[moi]
        print(f"{moi:4d} {p_s:12.4f} {p_h:12.4f}")


# ===========================================================================
# PROBLEM 3(a) - CRN for Z = X * log2(Y)
#
# Module 1 - Logarithm (W = log2(Y)):
#   b_l  --slow(1e-4)-->         a_l + b_l
#   a_l + 2y  --faster(1e4)-->   c_l + y' + a_l
#   2c_l  --faster(1e4)-->       c_l
#   a_l  --fast(1e2)-->          empty
#   y'   --medium(1)-->          y
#   c_l  --medium(1)-->          w
#
# Module 2 - Multiplication (Z = X * W):
#   x    --v.slow(1e-7)-->       a_m
#   a_m + w  --faster(1e4)-->    a_m + z' + w'
#   a_m  --fast(1e2)-->          empty
#   w'   --medium(1)-->          w
#   z'   --medium(1)-->          z
#
# Species indices: y=0, b_l=1, a_l=2, c_l=3, yp=4, w=5, x=6, a_m=7, zp=8, z=9, wp=10
# Initially: Y=Y0, b_l=1, X=X0, all others=0
# ===========================================================================

def _p3a_run(X0, Y0, max_steps=100000000, max_time=1e12):
    state = np.array([Y0, 1, 0, 0, 0, 0, X0, 0, 0, 0, 0], dtype=int)

    k_l1, k_l2, k_l3, k_l4, k_l5, k_l6 = 1e-4, 1e4, 1e4, 1e2, 1.0, 1.0
    k_m1, k_m2, k_m3, k_m4, k_m5       = 1e-7, 1e4, 1e2, 1.0, 1.0

    t = 0.0
    for _ in range(max_steps):
        y, b_l, a_l, c_l, yp, w, x, a_m, zp, z, wp = state
        if (y <= 1 and a_l == 0 and c_l == 0 and yp == 0 and
                x == 0 and a_m == 0 and zp == 0 and wp == 0):
            break

        props = np.zeros(11)
        props[0]  = k_l1 * b_l
        props[1]  = k_l2 * a_l * (y*(y-1)//2) if y >= 2 and a_l >= 1 else 0
        props[2]  = k_l3 * (c_l*(c_l-1)//2)   if c_l >= 2 else 0
        props[3]  = k_l4 * a_l
        props[4]  = k_l5 * yp
        props[5]  = k_l6 * c_l
        props[6]  = k_m1 * x
        props[7]  = k_m2 * a_m * w if a_m >= 1 and w >= 1 else 0
        props[8]  = k_m3 * a_m
        props[9]  = k_m4 * zp
        props[10] = k_m5 * wp

        a_total = np.sum(props)
        if a_total == 0:
            break
        tau = np.random.exponential(1.0 / a_total)
        t += tau
        if t > max_time:
            break

        r = np.random.random() * a_total
        cumsum = 0.0
        chosen = -1
        for i in range(11):
            cumsum += props[i]
            if r < cumsum:
                chosen = i
                break

        if   chosen == 0:  state[2] += 1
        elif chosen == 1:  state[0] -= 2; state[3] += 1; state[4] += 1
        elif chosen == 2:  state[3] -= 1
        elif chosen == 3:  state[2] -= 1
        elif chosen == 4:  state[4] -= 1; state[0] += 1
        elif chosen == 5:  state[3] -= 1; state[5] += 1
        elif chosen == 6:  state[6] -= 1; state[7] += 1
        elif chosen == 7:  state[5] -= 1; state[8] += 1; state[10] += 1
        elif chosen == 8:  state[7] -= 1
        elif chosen == 9:  state[8] -= 1; state[9] += 1
        elif chosen == 10: state[10] -= 1; state[5] += 1

    return state


def run_problem_3a():
    print("EE 5393 HW1 - Problem 3(a)")
    print("CRN for Z = X * log2(Y)")
    print()
    print("=" * 60)
    print("CRN Design (11 reactions, 11 species):")
    print("=" * 60)
    print()
    print("Module 1 - Logarithm (W = log2(Y)):")
    print("  b_l  --slow(1e-4)-->         a_l + b_l")
    print("  a_l + 2y  --faster(1e4)-->   c_l + y' + a_l")
    print("  2c_l  --faster(1e4)-->       c_l")
    print("  a_l  --fast(1e2)-->          empty")
    print("  y'   --medium(1)-->          y")
    print("  c_l  --medium(1)-->          w")
    print()
    print("Module 2 - Multiplication (Z = X * W):")
    print("  x    --v.slow(1e-7)-->       a_m")
    print("  a_m + w  --faster(1e4)-->    a_m + z' + w'")
    print("  a_m  --fast(1e2)-->          empty")
    print("  w'   --medium(1)-->          w")
    print("  z'   --medium(1)-->          z")
    print()
    print("Initially: Y=Y_0, X=X_0, B_l=1, all others=0.")
    print()

    test_cases = [(4, 8), (3, 16), (5, 4), (2, 32), (6, 2)]
    num_trials = 20

    print("=" * 60)
    print("Stochastic Simulation Results (20 trials each):")
    print("=" * 60)
    print()

    np.random.seed(42)
    for X0, Y0 in test_cases:
        expected = X0 * math.log2(Y0)
        z_vals = []
        t0 = time.time()
        for _ in range(num_trials):
            z_vals.append(int(_p3a_run(X0, Y0)[9]))
        print(f"X={X0}, Y={Y0}: Expected Z={expected:.0f}, "
              f"Mean={np.mean(z_vals):.2f}, Std={np.std(z_vals):.2f}  "
              f"[{time.time()-t0:.1f}s]")
        print(f"  Trials: {z_vals}")
        print()


# ===========================================================================
# PROBLEM 3(b) - CRN for Y = 2^(log2(X))
#
# Module 1 - Logarithm (W = log2(X)):
#   b_l  --slow(1e-4)-->         a_l + b_l
#   a_l + 2x  --faster(1e4)-->   c_l + x' + a_l
#   2c_l  --faster(1e4)-->       c_l
#   a_l  --fast(1e2)-->          empty
#   x'   --medium(1)-->          x
#   c_l  --medium(1)-->          w
#
# Module 2 - Exponentiation (Y = 2^W):
#   w    --v.slow(1e-7)-->       a_e
#   a_e + y  --faster(1e4)-->    a_e + 2y'
#   a_e  --fast(1e2)-->          empty
#   y'   --medium(1)-->          y
#
# Species indices: x=0, b_l=1, a_l=2, c_l=3, xp=4, w=5, y=6, a_e=7, yp=8
# Initially: X=X0, b_l=1, y=1, all others=0
# ===========================================================================

def _p3b_run(X0, max_steps=100000000, max_time=1e12):
    state = np.array([X0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=int)

    k_l1, k_l2, k_l3, k_l4, k_l5, k_l6 = 1e-4, 1e4, 1e4, 1e2, 1.0, 1.0
    k_e1, k_e2, k_e3, k_e4             = 1e-7, 1e4, 1e2, 1.0

    t = 0.0
    for _ in range(max_steps):
        x, b_l, a_l, c_l, xp, w, y, a_e, yp = state
        if (x <= 1 and a_l == 0 and c_l == 0 and xp == 0 and
                w == 0 and a_e == 0 and yp == 0):
            break

        props = np.zeros(10)
        props[0] = k_l1 * b_l
        props[1] = k_l2 * a_l * (x*(x-1)//2) if x >= 2 and a_l >= 1 else 0
        props[2] = k_l3 * (c_l*(c_l-1)//2)   if c_l >= 2 else 0
        props[3] = k_l4 * a_l
        props[4] = k_l5 * xp
        props[5] = k_l6 * c_l
        props[6] = k_e1 * w
        props[7] = k_e2 * a_e * y if a_e >= 1 and y >= 1 else 0
        props[8] = k_e3 * a_e
        props[9] = k_e4 * yp

        a_total = np.sum(props)
        if a_total == 0:
            break
        tau = np.random.exponential(1.0 / a_total)
        t += tau
        if t > max_time:
            break

        r = np.random.random() * a_total
        cumsum = 0.0
        chosen = -1
        for i in range(10):
            cumsum += props[i]
            if r < cumsum:
                chosen = i
                break

        if   chosen == 0: state[2] += 1
        elif chosen == 1: state[0] -= 2; state[3] += 1; state[4] += 1
        elif chosen == 2: state[3] -= 1
        elif chosen == 3: state[2] -= 1
        elif chosen == 4: state[4] -= 1; state[0] += 1
        elif chosen == 5: state[3] -= 1; state[5] += 1
        elif chosen == 6: state[5] -= 1; state[7] += 1
        elif chosen == 7: state[6] -= 1; state[8] += 2
        elif chosen == 8: state[7] -= 1
        elif chosen == 9: state[8] -= 1; state[6] += 1

    return state


def run_problem_3b():
    print("EE 5393 HW1 - Problem 3(b)")
    print("CRN for Y = 2^(log2(X))")
    print()
    print("=" * 60)
    print("CRN Design (10 reactions, 9 species):")
    print("=" * 60)
    print()
    print("Module 1 - Logarithm (W = log2(X)):")
    print("  b_l  --slow(1e-4)-->         a_l + b_l")
    print("  a_l + 2x  --faster(1e4)-->   c_l + x' + a_l")
    print("  2c_l  --faster(1e4)-->       c_l")
    print("  a_l  --fast(1e2)-->          empty")
    print("  x'   --medium(1)-->          x")
    print("  c_l  --medium(1)-->          w")
    print()
    print("Module 2 - Exponentiation (Y = 2^W):")
    print("  w    --v.slow(1e-7)-->       a_e")
    print("  a_e + y  --faster(1e4)-->    a_e + 2y'")
    print("  a_e  --fast(1e2)-->          empty")
    print("  y'   --medium(1)-->          y")
    print()
    print("Initially: X=X_0, Y=1, B_l=1, all others=0.")
    print("Result: Y = 2^(log2(X)) = X")
    print()

    test_cases = [2, 4, 8, 16, 32]
    num_trials = 20

    print("=" * 60)
    print("Stochastic Simulation Results (20 trials each):")
    print("=" * 60)
    print()

    np.random.seed(42)
    for X0 in test_cases:
        y_vals = []
        t0 = time.time()
        for _ in range(num_trials):
            y_vals.append(int(_p3b_run(X0)[6]))
        print(f"X={X0}: Expected Y={X0}, "
              f"Mean={np.mean(y_vals):.2f}, Std={np.std(y_vals):.2f}  "
              f"[{time.time()-t0:.1f}s]")
        print(f"  Trials: {y_vals}")
        print()


# ===========================================================================
# MENU
# ===========================================================================

PROBLEMS = {
    '1a': ('Problem 1(a) - Outcome probabilities from S=[110,26,55]', run_problem_1a),
    '1b': ('Problem 1(b) - Mean/variance after 7 steps from S=[9,8,7]',  run_problem_1b),
    '2':  ('Problem 2   - Lambda phage stealth vs hijack (MOI=1..10)',    run_problem_2),
    '3a': ('Problem 3(a) - CRN for Z = X * log2(Y)',                     run_problem_3a),
    '3b': ('Problem 3(b) - CRN for Y = 2^(log2(X))',                     run_problem_3b),
}


def main():
    # Allow direct invocation: python hw1.py 1a
    if len(sys.argv) > 1:
        key = sys.argv[1].lower()
        if key in PROBLEMS:
            PROBLEMS[key][1]()
        else:
            print(f"Unknown problem '{key}'. Valid options: {', '.join(PROBLEMS)}")
        return

    # Interactive menu
    print("=" * 60)
    print("EE 5393 Homework #1  —  Alex")
    print("=" * 60)
    print()
    print("Select a problem to run:")
    print()
    for key, (desc, _) in PROBLEMS.items():
        print(f"  [{key}]  {desc}")
    print()
    print("  [all]  Run all problems in sequence")
    print("  [q]    Quit")
    print()

    choice = input("Choice: ").strip().lower()
    print()

    if choice == 'q':
        return
    elif choice == 'all':
        for key, (desc, fn) in PROBLEMS.items():
            print()
            print("=" * 60)
            print(f"Running {desc}")
            print("=" * 60)
            print()
            fn()
            print()
    elif choice in PROBLEMS:
        PROBLEMS[choice][1]()
    else:
        print(f"Unknown choice '{choice}'.")


if __name__ == "__main__":
    main()
