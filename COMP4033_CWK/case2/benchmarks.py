"""
Benchmarking and results table runner for Case 2 
"""

from case2.mamdani_model import Case2FuzzyLogicSystem


def run_case2_benchmarks(test_patients_intervals):
    """Runs Final Case 2 (IT2, 21 rules) numeric & Jaccard linguistic table alongside the 64-rule variant."""
    fls_final = Case2FuzzyLogicSystem(rule_mode="compact_21")
    fls_64 = Case2FuzzyLogicSystem(rule_mode="exhaustive_64")

    print(f"\nCASE 2 FINAL MODEL: INTERVAL TYPE-2 MAMDANI ({len(fls_final.rules_log)} RULES)\n")
    print(
        f"{'Patient':<44} | {'Temp [C]':<11} | {'Head':<9} | {'Age':<11} | "
        f"{'Urgency*':<8} | {'Range*':<16} | {'Jaccard (L/M/H)*':<24} | {'64-Rule'}"
    )
    print("-" * 148)
    for name, t_int, h_int, a_int in test_patients_intervals:
        score_21, (u_min, u_max) = fls_final.evaluate_patient_interval(t_int, h_int, a_int)
        label, sims = fls_final.linguistic_output(t_int, h_int, a_int)
        score_64, _ = fls_64.evaluate_patient_interval(t_int, h_int, a_int)

        t_str = f"[{t_int[0]:.1f},{t_int[1]:.1f}]"
        h_str = f"[{h_int[0]:.1f},{h_int[1]:.1f}]"
        a_str = f"[{a_int[0]:.1f},{a_int[1]:.1f}]"
        u_range_str = f"[{u_min:>5.2f}, {u_max:>5.2f}]"
        sim_str = "/".join(f"{sims[k]:.2f}" for k in ("Low", "Medium", "High"))
        jac_str = f"{label:<6} ({sim_str})"

        print(
            f"{name:<44} | {t_str:<11} | {h_str:<9} | {a_str:<11} | "
            f"{score_21:>8.2f} | {u_range_str:<16} | {jac_str:<24} | {score_64:>7.2f}"
        )
    print("-" * 148)
    print("* Final Case 2 Model (21 Compact Rules | Numeric COS Centroid & Jaccard Linguistic Defuzzification)")

    return fls_final