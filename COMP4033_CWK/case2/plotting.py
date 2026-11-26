"""
Membership function plotting utility for Case 2 (Interval Type-2).
"""

from rules import setup_juzzy_path

setup_juzzy_path()

from juzzyPython.generic.Tuple import Tuple
from juzzyPython.generic.Plot import Plot


def plot_it2_membership_functions(fls):
    """Plots the 4 sets of Interval Type-2 membership functions (Upper & Lower MFs) using JuzzyPy's Plot utility."""
    plotter = Plot()

    variables = [
        ("IT2 Temperature Membership Functions (UMF & LMF)", fls.temp_in.getDomain(),
         [fls.temp_hypo, fls.temp_norm, fls.temp_fever, fls.temp_hyper]),
        ("IT2 Headache Severity Membership Functions (UMF & LMF)", fls.headache_in.getDomain(),
         [fls.head_none, fls.head_mild, fls.head_mod, fls.head_sev]),
        ("IT2 Patient Age Membership Functions (UMF & LMF)", fls.age_in.getDomain(),
         [fls.age_u5, fls.age_child, fls.age_adult, fls.age_older]),
        ("IT2 Urgency Output Membership Functions (UMF & LMF)", fls.urgency_out.getDomain(),
         [fls.urg_low, fls.urg_med, fls.urg_high]),
    ]

    for title, domain, mf_list in variables:
        plotter.figure()
        for it2_mf in mf_list:
            # Plot Upper MF and Lower MF so the Footprint of Uncertainty (FOU) is clearly visible
            plotter.plotMF(
                title,
                f"{it2_mf.getName()} (UMF)",
                it2_mf.getUMF(),
                500,
                domain,
                Tuple(0.0, 1.0),
                False
            )
            plotter.plotMF(
                title,
                f"{it2_mf.getName()} (LMF)",
                it2_mf.getLMF(),
                500,
                domain,
                Tuple(0.0, 1.0),
                False
            )
        plotter.legend()

    plotter.show()