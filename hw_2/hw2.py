"""
EE 5393 Homework #2 – Simulation and Verification
Alex, March 2026

Problems:
  1. Fibonacci CRN – design reactions for 12 Fibonacci steps
  2. Biquad Filter CRN – implement biquad with RGB delay methodology, 5 cycles

Usage:
  python hw2.py          # run everything
  python hw2.py 1        # Problem 1 only
  python hw2.py 2        # Problem 2 only
"""

import sys
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# PROBLEM 1: FIBONACCI CRN
# ============================================================
# Recurrence: F(n+1) = F(n) + F(n-1)
#
# CRN Design: Two-register system with RGB-phase sequencing.
# Each RGB cycle computes one Fibonacci step.
#
# Species:
#   B1, B2 : "Blue" phase values of register 1 and register 2
#             (these are the current values at start of each cycle)
#   T1, T2, T3 : intermediates used during the update
#   R_sum  : accumulates F(n) + F(n-1) during Red phase
#   R2     : carries old F(n) → new F(n-1) during Red phase
#   G_sum, G2 : Green-phase transits
#
# Reactions (one full RGB cycle):
#
#   ── Blue → Red phase (absence-indicator b fires slow reactions) ──
#   b + B1  –kslow→  T1 + T2       fanout B1 into T1 (→ new B2) + T2 (→ sum)
#   b + B2  –kslow→  T3             copy B2 to T3 (→ sum)
#   T1      –kfast→  R2             old F(n) becomes new F(n-1)
#   T2      –kfast→  R_sum          F(n) contributes to new sum
#   T3      –kfast→  R_sum          F(n-1) contributes to new sum
#
#   ── Red → Green phase ──
#   r + R_sum  –kslow→  G_sum
#   r + R2     –kslow→  G2
#
#   ── Green → Blue phase ──
#   g + G_sum  –kslow→  B1          B1 = F(n+1) for next cycle
#   g + G2     –kslow→  B2          B2 = F(n)   for next cycle
#
# Full absence-indicator + color-concentration-indicator reactions
# follow exactly the pattern in homework equations (2)–(5).
#
# For the ODE simulation below we use a SEQUENTIAL STEP model:
# we simulate the ODEs for one RGB cycle, read off B1 and B2,
# then start the next cycle. This matches the "run Aleae 12 times"
# approach described in the homework for the FIR/biquad filters.
# ============================================================

# Rate constants (same convention as FIR filter example)
K_SLOW = 1e-3
K_FAST = 1e3


def fibonacci_ode(t, y, kslow, kfast):
    """
    ODE for one Fibonacci RGB cycle.

    State vector y = [B1, B2, T1, T2, T3, R_sum, R2, G_sum, G2, b, r, g]
    where b, r, g are absence-indicator sources (held approximately constant
    by slow production and fast consumption when the matching color is present).

    For clarity we simulate the full chemistry including absence indicators.
    """
    B1, B2, T1, T2, T3, R_sum, R2, G_sum, G2, b, r, g = y

    # Absence indicator sources (S_b, S_r, S_g produce b, r, g slowly)
    S_b = S_r = S_g = 1.0   # constant source molecules (normalized)

    # ── Blue-to-Red reactions (require b) ──
    v_bB1  = kslow * b * B1    # b + B1  → T1 + T2
    v_bB2  = kslow * b * B2    # b + B2  → T3

    # Fast: intermediates T1,T2,T3 → Red phase species
    v_T1   = kfast * T1        # T1 → R2
    v_T2   = kfast * T2        # T2 → R_sum
    v_T3   = kfast * T3        # T3 → R_sum

    # ── Red-to-Green reactions (require r) ──
    v_rRs  = kslow * r * R_sum # r + R_sum → G_sum
    v_rR2  = kslow * r * R2    # r + R2    → G2

    # ── Green-to-Blue reactions (require g) ──
    v_gGs  = kslow * g * G_sum # g + G_sum → B1
    v_gG2  = kslow * g * G2    # g + G2    → B2

    # ── Absence indicator consumption (fast, by color concentration indicators) ──
    # b is consumed when Red species (R_sum, R2) are present
    v_b_consume = kfast * b * (R_sum + R2)
    # r is consumed when Green species (G_sum, G2) are present
    v_r_consume = kfast * r * (G_sum + G2)
    # g is consumed when Blue species (B1, B2) are present
    v_g_consume = kfast * g * (B1 + B2 + T1 + T2 + T3)  # any blue-phase species

    # ── Absence indicator production (slow, from sources) ──
    v_b_prod = kslow * S_b
    v_r_prod = kslow * S_r
    v_g_prod = kslow * S_g

    dB1    = -v_bB1           + v_gGs
    dB2    = -v_bB2           + v_gG2
    dT1    =  v_bB1           - v_T1
    dT2    =  v_bB1           - v_T2
    dT3    =  v_bB2           - v_T3
    dR_sum =  v_T2 + v_T3     - v_rRs
    dR2    =  v_T1            - v_rR2
    dG_sum =  v_rRs           - v_gGs
    dG2    =  v_rR2           - v_gG2
    db     =  v_b_prod        - v_b_consume
    dr     =  v_r_prod        - v_r_consume
    dg     =  v_g_prod        - v_g_consume

    return [dB1, dB2, dT1, dT2, dT3, dR_sum, dR2, dG_sum, dG2, db, dr, dg]


def fibonacci_crn_ode(start_a, start_b, n_steps=12, t_cycle=5000.0):
    """
    Simulate the Fibonacci CRN using ODEs for n_steps RGB cycles.
    Returns list of Fibonacci values observed after each cycle.

    start_a = F(1) (current),  start_b = F(0) (previous)
    """
    results = []
    B1, B2 = float(start_a), float(start_b)
    results.append(int(round(B2)))  # record F(0)
    results.append(int(round(B1)))  # record F(1)

    t_span = (0, t_cycle)
    t_eval = np.linspace(0, t_cycle, 2000)

    for step in range(n_steps):
        # Initial state: B1=current, B2=previous, intermediates=0, b=1,r=0,g=0
        y0 = [B1, B2, 0, 0, 0, 0, 0, 0, 0, 1.0, 0.0, 0.0]
        sol = solve_ivp(fibonacci_ode, t_span, y0,
                        args=(K_SLOW, K_FAST),
                        t_eval=t_eval,
                        method='RK45',
                        rtol=1e-8, atol=1e-10,
                        dense_output=False)
        # Read off final B1, B2
        B1_new = sol.y[0, -1]
        B2_new = sol.y[1, -1]
        B1, B2 = B1_new, B2_new
        results.append(int(round(B1_new)))

    return results


def fibonacci_exact(start_a, start_b, n_steps=12):
    """Exact integer Fibonacci by direct recurrence."""
    a, b = start_a, start_b
    seq = [b, a]
    for _ in range(n_steps):
        a, b = a + b, a
    return seq + [a + b for _ in range(n_steps)]  # recompute clean


def fibonacci_exact_clean(f0, f1, n_steps=12):
    """Return the sequence F(0)..F(n_steps) given F(0)=f0, F(1)=f1."""
    seq = [f0, f1]
    for i in range(n_steps - 1):
        seq.append(seq[-1] + seq[-2])
    return seq  # length = n_steps + 1


def run_problem1():
    print("=" * 60)
    print("PROBLEM 1: FIBONACCI CRN")
    print("=" * 60)

    cases = [(0, 1), (3, 7)]
    for f0, f1 in cases:
        seq = fibonacci_exact_clean(f0, f1, n_steps=12)
        print(f"\nStarting values ({f0}, {f1}):")
        print(f"  Steps 0-12: {seq}")
        print(f"  Step 12 (final): {seq[-1]}")

    print()
    print("─" * 60)
    print("CRN verification via ODE simulation (one RGB cycle = one step)")
    print("─" * 60)
    for f0, f1 in cases:
        print(f"\nODE simulation starting ({f0}, {f1}):")
        ode_seq = fibonacci_crn_ode(f1, f0, n_steps=11, t_cycle=3000.0)
        exact    = fibonacci_exact_clean(f0, f1, n_steps=12)
        print(f"  Exact:    {exact}")
        print(f"  ODE sim:  {ode_seq[:13]}")
        ok = all(abs(o - e) < 0.5 for o, e in zip(ode_seq, exact))
        print(f"  Match: {'✓ PASS' if ok else '✗ FAIL'}")


# ============================================================
# PROBLEM 2: BIQUAD FILTER CRN
# ============================================================
# Figure 2a biquad equations:
#
#   A[n] = X[n] + (1/8)*A[n-1] + (1/8)*A[n-2]
#   Y[n] = (1/8) * A[n]
#
# Here A is the input to the first delay cell.
# The two delay cells store A[n-1] and A[n-2].
#
# CRN Design (extending FIR filter reactions (1)-(5)):
#
# Two delay cells: (R1,G1,B1) for A[n-1]  and  (R2,G2,B2) for A[n-2]
#
# The 1/8 coefficient requires THREE cascaded bimolecular halvings:
#   2C   –kfast→  C2        (C2 = C/2)
#   2C2  –kfast→  C4        (C4 = C/4)
#   2C4  –kfast→  R_fbk     (R_fbk = C/8)
#
# Reaction groups:
#
# GROUP 1 – Input fanout + 1/8 scaling of delays (Blue phase):
#   g + X   –kslow→  Ax + Cx            fanout X: Ax→ A_new, Cx→ Y
#   g + B1  –kslow→  Ab1 + Cb1 + Db1   fanout B1 (= A[n-1]):
#                                         Ab1→ delay2 input, Cb1→ 1/8 fbk, Db1→ 1/8 fwd
#   g + B2  –kslow→  Cb2 + Db2          fanout B2 (= A[n-2]):
#                                         Cb2→ 1/8 fbk, Db2→ 1/8 fwd
#   Triple halving for 1/8 feedback from B1:
#   2Cb1 –kfast→ Cb1h,  2Cb1h –kfast→ Cb1q,  2Cb1q –kfast→ R_fbk1
#   2Cb2 –kfast→ Cb2h,  2Cb2h –kfast→ Cb2q,  2Cb2q –kfast→ R_fbk2
#   Triple halving for 1/8 forward contribution to Y:
#   2Db1 –kfast→ Db1h,  2Db1h –kfast→ Db1q,  2Db1q –kfast→ R_fwd1
#   2Db2 –kfast→ Db2h,  2Db2h –kfast→ Db2q,  2Db2q –kfast→ R_fwd2
#   2Cx  –kfast→ Cxh,   2Cxh  –kfast→ Cxq,   2Cxq  –kfast→ R_fwdX
#   Ax   –kfast→ R_Anew            X contributes directly (coeff 1) to A_new
#   R_fbk1 + R_fbk2 –kfast→ R_Anew (feedback from both delays)
#
# GROUP 2 – Delay cell 2 input (from A[n-1] copy):
#   Ab1  –kfast→ R2_new   (B1 copy becomes input to second delay cell)
#
# GROUP 3 – Delay sequencing (same as FIR filter groups (2)-(4)):
#   (same absence-indicator mechanism for R→G→B→R cycling for both cells)
#
# GROUP 4 – Output collection:
#   R_fwdX + R_fwd1 + R_fwd2 → Y  (sum of scaled X, C, E → output Y)
#
# ============================================================

def biquad_math(X_seq, v1=0.0, v2=0.0):
    """
    Compute biquad filter output directly from the recurrence.

    From Figure 2a (all 1/8 coefficients symmetric):
      A[n] = X[n] + (1/8)*v1[n] + (1/8)*v2[n]   <- input to delay cell 1
      Y[n] = (1/8)*X[n] + (1/8)*v1[n] + (1/8)*v2[n]  <- output (all equal 1/8)
      v1[n+1] = A[n]        <- delay cell 1 stores A[n]
      v2[n+1] = v1[n]       <- delay cell 2 is pure delay of cell 1

    Note: Y[n] = A[n] - (7/8)*X[n], and A[n] = (7/8)*X[n] + Y[n].
    The state variable A (= 8*Y for X=0) is what cycles through the delay cells.

    Returns list of (A[n], Y[n], v1_new, v2_new) for each input in X_seq.
    """
    results = []
    for X in X_seq:
        A = X + (1/8)*v1 + (1/8)*v2
        Y = (1/8)*X + (1/8)*v1 + (1/8)*v2
        results.append((A, Y))
        v2, v1 = v1, A
    return results


def biquad_ode(t, y, kslow, kfast, X_val):
    """
    ODE system for ONE biquad RGB cycle.

    Species: [B1, B2, Ax, Cx, Ab1, Cb1, Db1, Ab2, Cb2, Db2,
              Cb1h, Cb1q, Cb2h, Cb2q, Db1h, Db1q, Db2h, Db2q, Cxh, Cxq,
              R_fbk1, R_fbk2, R_fwd1, R_fwd2, R_fwdX, R_Anew, R2_new,
              G1, G2, G_Anew, G2_new,
              Y_out,
              b, r, g]

    B1 = A[n-1] (blue, delay cell 1 output)
    B2 = A[n-2] (blue, delay cell 2 output)
    """
    (B1, B2, Ax, Cx, Ab1, Cb1, Db1, Ab2, Cb2, Db2,
     Cb1h, Cb1q, Cb2h, Cb2q, Db1h, Db1q, Db2h, Db2q, Cxh, Cxq,
     R_fbk1, R_fbk2, R_fwd1, R_fwd2, R_fwdX, R_Anew, R2_new,
     G1, G2, G_Anew, G2_new,
     Y_out,
     b, r, g) = y

    S_b = S_r = S_g = 1.0
    X = X_val  # external input (constant during this cycle)

    # ── Blue-phase fanout reactions (require b) ──
    v_X     = kslow * b * X        # g + X → Ax + Cx   (using b as gate here for blue phase)
    v_B1    = kslow * b * B1       # b + B1 → Ab1 + Cb1 + Db1
    v_B2    = kslow * b * B2       # b + B2 → Cb2 + Db2

    # Fast: triple halving for 1/8 coefficients
    # From Cb1 (feedback path, delay 1)
    v_Cb1_half = kfast * Cb1 * max(Cb1 - 1, 0) / 2.0   # 2Cb1 → Cb1h
    v_Cb1h_half = kfast * Cb1h * max(Cb1h - 1, 0) / 2.0
    v_Cb1q_to_R = kfast * Cb1q * max(Cb1q - 1, 0) / 2.0
    # From Cb2 (feedback path, delay 2)
    v_Cb2_half = kfast * Cb2 * max(Cb2 - 1, 0) / 2.0
    v_Cb2h_half = kfast * Cb2h * max(Cb2h - 1, 0) / 2.0
    v_Cb2q_to_R = kfast * Cb2q * max(Cb2q - 1, 0) / 2.0
    # From Db1 (forward/output path, delay 1)
    v_Db1_half = kfast * Db1 * max(Db1 - 1, 0) / 2.0
    v_Db1h_half = kfast * Db1h * max(Db1h - 1, 0) / 2.0
    v_Db1q_to_R = kfast * Db1q * max(Db1q - 1, 0) / 2.0
    # From Db2 (forward/output path, delay 2)
    v_Db2_half = kfast * Db2 * max(Db2 - 1, 0) / 2.0
    v_Db2h_half = kfast * Db2h * max(Db2h - 1, 0) / 2.0
    v_Db2q_to_R = kfast * Db2q * max(Db2q - 1, 0) / 2.0
    # From Cx (forward/output path, X input)
    v_Cx_half  = kfast * Cx * max(Cx - 1, 0) / 2.0
    v_Cxh_half = kfast * Cxh * max(Cxh - 1, 0) / 2.0
    v_Cxq_to_R = kfast * Cxq * max(Cxq - 1, 0) / 2.0

    # Direct transfer: Ax → R_Anew  (X contributes fully to A_new)
    v_Ax_to_R  = kfast * Ax
    # Direct transfer: Ab1 → R2_new  (B1 copy → input to delay 2)
    v_Ab1_to_R2 = kfast * Ab1
    # Feedback additions to R_Anew
    v_fbk1_to_Anew = kfast * R_fbk1
    v_fbk2_to_Anew = kfast * R_fbk2

    # ── Red-to-Green (require r) ──
    v_rAnew  = kslow * r * R_Anew
    v_rR2    = kslow * r * R2_new
    v_rFwdX  = kslow * r * R_fwdX
    v_rFwd1  = kslow * r * R_fwd1
    v_rFwd2  = kslow * r * R_fwd2

    # ── Green-to-Blue (require g) ──
    v_gAnew = kslow * g * G_Anew   # G_Anew → B1 (new A[n] goes to delay 1)
    v_gR2   = kslow * g * G2_new   # G2_new → B2 (new A[n-1] goes to delay 2)
    v_gY    = kslow * g * (G1 + G2) # G1 + G2 → Y_out (output collection)

    # Absence indicators
    red_signal   = R_Anew + R2_new + R_fwdX + R_fwd1 + R_fwd2 + R_fbk1 + R_fbk2
    green_signal = G1 + G2 + G_Anew + G2_new
    blue_signal  = B1 + B2 + Ax + Cx + Ab1 + Cb1 + Db1 + Ab2 + Cb2 + Db2

    v_b_prod = kslow * S_b
    v_r_prod = kslow * S_r
    v_g_prod = kslow * S_g
    v_b_con  = kfast * b * red_signal
    v_r_con  = kfast * r * green_signal
    v_g_con  = kfast * g * blue_signal

    dB1    = -v_B1                    + v_gAnew
    dB2    = -v_B2                    + v_gR2
    dAx    =  v_X                     - v_Ax_to_R
    dCx    =  v_X                     - v_Cx_half
    dAb1   =  v_B1                    - v_Ab1_to_R2
    dCb1   =  v_B1                    - v_Cb1_half
    dDb1   =  v_B1                    - v_Db1_half
    dAb2   =  0  # not used
    dCb2   =  v_B2                    - v_Cb2_half
    dDb2   =  v_B2                    - v_Db2_half

    dCb1h  =  v_Cb1_half              - v_Cb1h_half
    dCb1q  =  v_Cb1h_half             - v_Cb1q_to_R
    dCb2h  =  v_Cb2_half              - v_Cb2h_half
    dCb2q  =  v_Cb2h_half             - v_Cb2q_to_R
    dDb1h  =  v_Db1_half              - v_Db1h_half
    dDb1q  =  v_Db1h_half             - v_Db1q_to_R
    dDb2h  =  v_Db2_half              - v_Db2h_half
    dDb2q  =  v_Db2h_half             - v_Db2q_to_R
    dCxh   =  v_Cx_half               - v_Cxh_half
    dCxq   =  v_Cxh_half              - v_Cxq_to_R

    dR_fbk1  = v_Cb1q_to_R            - v_fbk1_to_Anew
    dR_fbk2  = v_Cb2q_to_R            - v_fbk2_to_Anew
    dR_fwd1  = v_Db1q_to_R            - v_rFwd1
    dR_fwd2  = v_Db2q_to_R            - v_rFwd2
    dR_fwdX  = v_Cxq_to_R             - v_rFwdX
    dR_Anew  = v_Ax_to_R + v_fbk1_to_Anew + v_fbk2_to_Anew - v_rAnew
    dR2_new  = v_Ab1_to_R2            - v_rR2

    dG1      = v_rFwd1 + v_rFwd2      - v_gY
    dG2      = v_rFwdX                - v_gY
    dG_Anew  = v_rAnew                - v_gAnew
    dG2_new  = v_rR2                  - v_gR2

    dY_out   = v_gY

    db = v_b_prod - v_b_con
    dr = v_r_prod - v_r_con
    dg = v_g_prod - v_g_con

    return [dB1, dB2, dAx, dCx, dAb1, dCb1, dDb1, dAb2, dCb2, dDb2,
            dCb1h, dCb1q, dCb2h, dCb2q, dDb1h, dDb1q, dDb2h, dDb2q, dCxh, dCxq,
            dR_fbk1, dR_fbk2, dR_fwd1, dR_fwd2, dR_fwdX, dR_Anew, dR2_new,
            dG1, dG2, dG_Anew, dG2_new,
            dY_out,
            db, dr, dg]


def run_problem2():
    print("=" * 60)
    print("PROBLEM 2: BIQUAD FILTER CRN")
    print("=" * 60)

    X_inputs = [100, 5, 500, 20, 250]

    # ── Exact mathematical result ──
    print("\nExact biquad filter output (direct recurrence):")
    print(f"  Equations: A[n] = X[n] + (1/8)*A[n-1] + (1/8)*A[n-2]")
    print(f"             Y[n] = (1/8)*A[n]")
    print()
    print(f"  {'Cycle':>5}  {'X':>6}  {'A[n]':>10}  {'Y[n]':>10}")
    print(f"  {'─'*5}  {'─'*6}  {'─'*10}  {'─'*10}")

    results = biquad_math(X_inputs)
    for i, ((A, Y), X) in enumerate(zip(results, X_inputs)):
        print(f"  {i:>5}  {X:>6}  {A:>10.4f}  {Y:>10.4f}")
    print()
    print("  Note: A[n] = X + (1/8)*v1 + (1/8)*v2  (stored in delay cell 1)")
    print("        Y[n] = (1/8)*X + (1/8)*v1 + (1/8)*v2  (all equal 1/8 coefficients)")
    print("        v1 = A[n-1], v2 = A[n-2]")

    print()
    print("Reaction design summary (extending FIR filter to biquad):")
    print("""
  The biquad CRN extends the moving-average FIR filter with:
  1. TWO delay cells (R1G1B1 for A[n-1], R2G2B2 for A[n-2])
  2. THREE cascaded bimolecular halvings for each 1/8 coefficient:
       2C  → C2  (C2 = C/2)
       2C2 → C4  (C4 = C/4)
       2C4 → Rfbk (Rfbk = C/8)
  3. Input X fans out to: (a) full contribution to A_new (coeff 1),
     (b) 1/8 contribution to output Y
  4. Each delay cell output fans out to: (a) 1/8 feedback to A_new,
     (b) 1/8 feedforward to Y, (c) full value to next delay cell (delay 2 only)
  5. Absence indicators r,g,b and sources Sr,Sg,Sb unchanged from FIR design.
    """)


# ============================================================
# PLOTTING
# ============================================================

def make_plots():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Fibonacci sequences
    ax = axes[0]
    for f0, f1, color, label in [(0, 1, 'steelblue', 'Start (0,1)'),
                                   (3, 7, 'tomato',    'Start (3,7)')]:
        seq = fibonacci_exact_clean(f0, f1, 12)
        ax.plot(range(13), seq, 'o-', color=color, label=label)
    ax.set_title('Problem 1: Fibonacci Sequences (12 steps)')
    ax.set_xlabel('Step n')
    ax.set_ylabel('F(n)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Biquad filter output
    ax = axes[1]
    X_inputs = [100, 5, 500, 20, 250]
    results = biquad_math(X_inputs)
    cycles = list(range(5))
    Y_vals = [r[1] for r in results]
    ax.bar(cycles, X_inputs, alpha=0.4, label='Input X', color='steelblue')
    ax.plot(cycles, Y_vals, 'ro-', linewidth=2, markersize=8, label='Output Y')
    ax.set_title('Problem 2: Biquad Filter (5 cycles)')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hw2_plots.png', dpi=150)
    print("\nPlots saved to hw2_plots.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if mode in ('1', 'all'):
        run_problem1()

    if mode in ('2', 'all'):
        print()
        run_problem2()

    if mode == 'all':
        make_plots()
