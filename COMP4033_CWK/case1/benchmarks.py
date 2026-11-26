"""
Benchmarking and comparison table runner for Case 1 (Final TSK + Mamdani Comparisons).
"""

from case1.tsk_model import Case1TSKFuzzyLogicSystem
from case1.mamdani_model import Case1FuzzyLogicSystem


def run_case1_benchmarks(test_patients):
    """Runs Final Case 1 (TSK) results alongside Mamdani, Rule-Base, Defuzz, MF, and Operator comparisons."""
    # TSK is the final model for Case 1
    tsk_final = Case1TSKFuzzyLogicSystem(mf_style="piecewise", operator_family="min_max", rule_mode="compact_21")
    print(f"CASE 1 FINAL MODEL: ZEROTH-ORDER TSK ({len(tsk_final.rules_log)} RULES | Piecewise | Min/Max)\n")

    # Prints final model results for Case 1 TSK
    print("1. FINAL CASE 1 MODEL RESULTS (Zeroth-Order TSK)")
    print(f"{'Patient':<47} | {'Temp':<5} | {'Head':<5} | {'Age':<5} | {'Urgency (0-100)'}")
    print("-" * 85)
    for name, t, h, a in test_patients:
        score = tsk_final.evaluate_patient(t, h, a)
        print(f"{name:<47} | {t:>5.1f} | {h:>5.1f} | {a:>5.1f} | {score:>10.2f}")
    print("-" * 85)

    # comparing TSK vs Mamdani, 21 vs 64 rules, MF shapes, and defuzzification methods
    tsk_64 = Case1TSKFuzzyLogicSystem(mf_style="piecewise", operator_family="min_max", rule_mode="exhaustive_64")
    mam_21 = Case1FuzzyLogicSystem(mf_style="piecewise", operator_family="min_max", rule_mode="compact_21")
    mam_64 = Case1FuzzyLogicSystem(mf_style="piecewise", operator_family="min_max", rule_mode="exhaustive_64")

    print("\n2. MODEL & RULE BASE COMPARISON (TSK vs Mamdani | 21 vs 64 Rules)")
    print(f"{'Patient':<47} | {'TSK (21)*':<10} | {'TSK (64)':<10} | {'Mamdani (21)':<12} | {'Mamdani (64)'}")
    print("-" * 102)
    for name, t, h, a in test_patients:
        s_t21 = tsk_final.evaluate_patient(t, h, a)
        s_t64 = tsk_64.evaluate_patient(t, h, a)
        s_m21 = mam_21.evaluate_patient(t, h, a)
        s_m64 = mam_64.evaluate_patient(t, h, a)
        print(f"{name:<47} | {s_t21:>10.2f} | {s_t64:>10.2f} | {s_m21:>12.2f} | {s_m64:>12.2f}")
    print("-" * 102)
    print("* Final Case 1 Model")

    # TSK Architecture Comparison (MF Shapes & Operators)
    tsk_gauss = Case1TSKFuzzyLogicSystem(mf_style="gaussian", operator_family="min_max", rule_mode="compact_21")
    tsk_hybrid = Case1TSKFuzzyLogicSystem(mf_style="hybrid", operator_family="min_max", rule_mode="compact_21")
    tsk_prod = Case1TSKFuzzyLogicSystem(mf_style="piecewise", operator_family="product", rule_mode="compact_21")

    print("\n3. CASE 1 TSK ARCHITECTURE COMPARISON: MF SHAPES & OPERATORS (21 Rules)")
    print(f"{'Patient':<47} | {'Piecewise (Min)*':<16} | {'Gaussian (Min)':<14} | {'Hybrid (Min)':<12} | {'Piecewise (Prod)'}")
    print("-" * 113)
    for name, t, h, a in test_patients:
        s_pw = tsk_final.evaluate_patient(t, h, a)
        s_ga = tsk_gauss.evaluate_patient(t, h, a)
        s_hy = tsk_hybrid.evaluate_patient(t, h, a)
        s_pr = tsk_prod.evaluate_patient(t, h, a)
        print(f"{name:<47} | {s_pw:>16.2f} | {s_ga:>14.2f} | {s_hy:>12.2f} | {s_pr:>16.2f}")
    print("-" * 113)

    # Mamdani Defuzzification Comparison (21 Rules)
    print("\n4. MAMDANI DEFUZZIFICATION COMPARISON (Centroid vs Height vs MoM vs Bisector | 21 Rules)")
    print(f"{'Patient':<47} | {'Centroid':<8} | {'Height':<8} | {'MoM':<8} | {'Bisector':<8}")
    print("-" * 93)
    for name, t, h, a in test_patients:
        res = mam_21.evaluate_all_defuzz_methods(t, h, a)
        print(f"{name:<47} | {res['Centroid']:>8.2f} | {res['Height']:>8.2f} | {res['MoM']:>8.2f} | {res['Bisector']:>8.2f}")
    print("-" * 93)

    return tsk_final, mam_21