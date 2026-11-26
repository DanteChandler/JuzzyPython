"""
Tests how well Case 2 handles uncertainty:
1. Interval Widening: Checks if wider input ranges lead to wider urgency ranges.
2. Singleton Check: Checks if exact numbers [x, x] in Case 2 match Case 1.
"""

import sys
from pathlib import Path

# Lets us run this file directly from the tests folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from case1.tsk_model import Case1TSKFuzzyLogicSystem
from case1.mamdani_model import Case1FuzzyLogicSystem
from case2.mamdani_model import Case2FuzzyLogicSystem
from tests.test_patients import TEST_PATIENTS_NUMERIC


def run_interval_widening_test(fls_case2):
    """Tests Patient 2 with wider and wider input ranges around the same middle value."""
    # Middle values for Patient 2 (Temp, Headache, Age)
    t_mid, h_mid, a_mid = 38.6, 5.0, 8.0
    deltas = [0.0, 0.1, 0.5, 1.5, 3.0]

    print("1. INTERVAL WIDENING TEST (Patient 2 Midpoint: Temp=38.6C, Head=5.0, Age=8.0)")
    print(
        f"{'Half-Width':<10} | {'Temp [C]':<13} | {'Head [0-10]':<11} | {'Age [Yrs]':<11} | "
        f"{'Urgency':<7} | {'Range [y_l, y_r]':<16} | {'Spread':<6} | {'Jaccard (L/M/H)'}"
    )
    print("-" * 124)

    for d in deltas:
        t_int = (max(0.0, t_mid - d), min(50.0, t_mid + d))
        h_int = (max(0.0, h_mid - d), min(10.0, h_mid + d))
        a_int = (max(0.0, a_mid - d), min(130.0, a_mid + d))

        score, (u_min, u_max) = fls_case2.evaluate_patient_interval(t_int, h_int, a_int)
        label, sims = fls_case2.linguistic_output(t_int, h_int, a_int)
        spread = u_max - u_min

        t_str = f"[{t_int[0]:.1f}, {t_int[1]:.1f}]"
        h_str = f"[{h_int[0]:.1f}, {h_int[1]:.1f}]"
        a_str = f"[{a_int[0]:.1f}, {a_int[1]:.1f}]"
        r_str = f"[{u_min:>5.2f}, {u_max:>5.2f}]"
        sim_str = "/".join(f"{sims[k]:.2f}" for k in ("Low", "Medium", "High"))
        jac_str = f"{label:<6} ({sim_str})"

        print(
            f"±{d:<9.1f} | {t_str:<13} | {h_str:<11} | {a_str:<11} | "
            f"{score:>7.2f} | {r_str:<16} | {spread:>6.2f} | {jac_str}"
        )
    print("-" * 124)


def run_singleton_limit_check(fls_tsk, fls_mam, fls_case2):
    """Tests Case 2 with exact numbers [x, x] and compares them to Case 1."""
    print("\n2. SINGLETON LIMIT CHECK (Case 1 TSK & Mamdani vs Case 2 Zero-Width [x, x] Intervals)")
    print(
        f"{'Patient':<44} | {'C1 TSK':<7} | {'C1 Mam':<7} | {'C2 [x,x]':<8} | "
        f"{'C2 FOU Range':<16} | {'C2 Jaccard (L/M/H)'}"
    )
    print("-" * 122)

    for name, t, h, a in TEST_PATIENTS_NUMERIC:
        tsk_score = fls_tsk.evaluate_patient(t, h, a)
        mam_score = fls_mam.evaluate_patient(t, h, a)
        c2_score, (u_min, u_max) = fls_case2.evaluate_patient_interval((t, t), (h, h), (a, a))
        label, sims = fls_case2.linguistic_output((t, t), (h, h), (a, a))

        r_str = f"[{u_min:>5.2f}, {u_max:>5.2f}]"
        sim_str = "/".join(f"{sims[k]:.2f}" for k in ("Low", "Medium", "High"))
        jac_str = f"{label:<6} ({sim_str})"

        print(
            f"{name:<44} | {tsk_score:>7.2f} | {mam_score:>7.2f} | {c2_score:>8.2f} | "
            f"{r_str:<16} | {jac_str}"
        )
    print("-" * 122)


if __name__ == "__main__":
    fls_tsk = Case1TSKFuzzyLogicSystem(mf_style="piecewise", operator_family="min_max", rule_mode="compact_21")
    fls_mam = Case1FuzzyLogicSystem(mf_style="piecewise", operator_family="min_max", rule_mode="compact_21")
    fls_c2 = Case2FuzzyLogicSystem(rule_mode="compact_21")

    run_interval_widening_test(fls_c2)
    run_singleton_limit_check(fls_tsk, fls_mam, fls_c2)