"""
Creates side-by-side heatmaps showing how the Case 1 Mamdani urgency score 
changes with temperature and headache severity for set ages.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Allow running this script directly from inside tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from case1.mamdani_model import Case1FuzzyLogicSystem


def plot_control_surfaces(fls_case1, ages=(25.0, 70.0), temp_range=(33.0, 42.0), resolution=25):
    """Generates side-by-side 2D filled contour control surfaces for the specified fixed ages."""
    # Set discretization level for the output variable to ensure smooth surfaces
    fls_case1.urgency_out.setDiscretisationLevel(50)

    temps = np.linspace(temp_range[0], temp_range[1], resolution)
    headaches = np.linspace(0.0, 10.0, resolution)
    T, H = np.meshgrid(temps, headaches)

    fig, axes = plt.subplots(1, len(ages), figsize=(13, 5), sharey=True)
    if len(ages) == 1:
        axes = [axes]

    for ax, age in zip(axes, ages):
        print(f"Computing surface for Age = {int(age)}...")
        Z = np.zeros_like(T)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = fls_case1.evaluate_patient(T[i, j], H[i, j], age)

        contour = ax.contourf(T, H, Z, levels=20, cmap="RdYlGn_r", vmin=0, vmax=100)
        ax.set_title(f"Urgency Control Surface (Age = {int(age)})")
        ax.set_xlabel("Temperature (°C)")

        # Mark key clinical thresholds
        for thresh in (35.0, 38.0, 40.0):
            ax.axvline(thresh, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    axes[0].set_ylabel("Headache Severity (0-10)")
    fig.colorbar(contour, ax=axes, label="Urgency Score (0-100)", pad=0.02)
    plt.suptitle(f"Case 1 Control Surfaces ({fls_case1.mf_style.capitalize()} MFs | {fls_case1.operator_family})", fontsize=13)
    plt.show()


if __name__ == "__main__":
    fls_c1 = Case1FuzzyLogicSystem(mf_style="piecewise", operator_family="min_max")
    plot_control_surfaces(fls_c1, ages=(25.0, 70.0))