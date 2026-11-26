"""
Membership function plotting utility for Case 1 (works for both TSK and Mamdani).
"""

from rules import setup_juzzy_path

setup_juzzy_path()

from juzzyPython.generic.Tuple import Tuple
from juzzyPython.generic.Plot import Plot


def plot_t1_membership_functions(fls):
    """Plots the input and output membership functions for a given fuzzy logic system (TSK or Mamdani)."""
    plotter = Plot()

    variables = [
        (f"Temperature MFs ({fls.mf_style})", fls.temp_in.getDomain(),
         [fls.temp_hypo, fls.temp_norm, fls.temp_fever, fls.temp_hyper]),
        (f"Headache Severity MFs ({fls.mf_style})", fls.headache_in.getDomain(),
         [fls.head_none, fls.head_mild, fls.head_mod, fls.head_sev]),
        (f"Patient Age MFs ({fls.mf_style})", fls.age_in.getDomain(),
         [fls.age_u5, fls.age_child, fls.age_adult, fls.age_older]),
    ]

    # Include output MFs only if plotting a Mamdani system (TSK uses crisp output constants)
    if hasattr(fls, "urgency_out"):
        variables.append(
            (f"Urgency Output MFs ({fls.mf_style})", fls.urgency_out.getDomain(),
             [fls.urg_low, fls.urg_med, fls.urg_high])
        )

    for title, domain, mf_list in variables:
        plotter.figure()
        for mf in mf_list:
            plotter.plotMF(
                title,
                mf.getName(),
                mf,
                500,
                domain,
                Tuple(0.0, 1.0),
                False
            )
        plotter.legend()

    plotter.show()