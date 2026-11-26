"""
Main entry point for the Fuzzy Inference Systems.

FINAL MODELS:
- Case 1 (Numeric):         Zeroth-Order TSK (Piecewise MFs | Min/Max | 21 Compact Rules)
- Case 2 (Interval): Interval Type-2 Mamdani (Piecewise IT2 | Min/Max | 21 Compact Rules)
"""

from tests.test_patients import TEST_PATIENTS_NUMERIC, TEST_PATIENTS_INTERVALS
from case1.benchmarks import run_case1_benchmarks
from case1.plotting import plot_t1_membership_functions
from case2.benchmarks import run_case2_benchmarks
from case2.plotting import plot_it2_membership_functions


if __name__ == "__main__":
    # Run Case 1 (Final TSK + Mamdani/Rule/MF/Operator/Defuzz Comparisons)
    tsk_final, mam_comp = run_case1_benchmarks(TEST_PATIENTS_NUMERIC)

    # Run Case 2 (Final IT2 Mamdani + 21 vs 64 Rule Comparison)
    it2_final = run_case2_benchmarks(TEST_PATIENTS_INTERVALS)

    # Optional: Print final 21-rule table
    ans = input(f"\nPrint all {len(tsk_final.rules_log)} final rules to terminal? [y/n]: ").strip().lower()
    if ans == "y":
        print("\n--- CASE 1 FINAL (TSK) RULES ---")
        for line in tsk_final.rules_log:
            print(line)

    # Optional: Launch Case 1 (Final TSK) Membership Function plots
    plot_t1 = input("Display Case 1 (Final TSK) Membership Function Plots? [y/n]: ").strip().lower()
    if plot_t1 == "y":
        plot_t1_membership_functions(tsk_final)

    # Optional: Launch Case 2 (Final IT2) Membership Function plots
    plot_it2 = input("Display Case 2 (Final IT2) Membership Function Plots? [y/n]: ").strip().lower()
    if plot_it2 == "y":
        plot_it2_membership_functions(it2_final)