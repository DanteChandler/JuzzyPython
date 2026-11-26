# COMP4033 Medical Assessment Fuzzy Inference Systems

Medical triage fuzzy logic systems evaluating Temperature, Headache Severity, and Patient Age to output an Urgency score (0-100) and linguistic level (Low, Medium, High).

- Case 1 Final Model (Numeric Inputs): Zeroth-Order TSK (Piecewise MFs, Min/Max operators, 21 compact rules)
- Case 2 Final Model (Interval Inputs): Interval Type-2 Mamdani (Piecewise IT2 UMF/LMF, Min/Max operators, 21 compact rules, Numeric COS Centroid and Jaccard Linguistic Defuzzification)

## Prerequisites

- Python 3.x (Anaconda recommended)
- Keep the JuzzyPython folder directly inside the COMP4033_CWK root directory
- Install required packages:

```bash
pip install numpy matplotlib
```

## How to Run

Open a terminal in the COMP4033_CWK root directory.

### 1. Run Final Models, Comparison Tables & Plots

```bash
python main.py
```

This prints:
- Case 1 Final Model (TSK) results
- TSK vs. Mamdani and 21-rule vs. 64-rule comparisons
- MF shape (Piecewise, Gaussian, Hybrid) and operator (Min/Max vs. Product) comparisons
- Mamdani defuzzification comparison (Centroid, Height, MoM, Bisector)
- Case 2 Final Model (IT2) numeric urgency, interval range, Jaccard linguistic output (L/M/H), and 64-rule comparison

At the end, the terminal prompts (y/n) to print the 21-rule base and display the Case 1 and Case 2 membership function plots.

### 2. Run Uncertainty & FOU Sensitivity Tests

```bash
python tests/test_uncertainty.py
```

Runs two tests on Case 2:
- Interval Widening Test: evaluates Patient 2 across wider input intervals (±0.0 to ±3.0) to check output range expansion and Jaccard similarity shifts
- Singleton Limit Check: feeds exact [x, x] intervals into Case 2 and compares them against Case 1 (TSK and Mamdani) crisp outputs

### 3. Run 2D Control Surface Plots

```bash
python tests/test_control_surfaces.py
```

Generates side-by-side Urgency heatmaps (Temperature vs. Headache Severity) for Age 25 and Age 70, with dashed lines at 35°C, 38°C, and 40°C.

## Project Structure

- main.py - Entry point for running final models, benchmarks, rules, and plots
- rules.py - Shared 21-rule (final) and 64-rule (comparison) bases plus JuzzyPython path setup
- case1/ - Final TSK model (tsk_model.py), comparative Mamdani model (mamdani_model.py), benchmarks, and plotter
- case2/ - Final IT2 Mamdani model with Jaccard similarity (mamdani_model.py), benchmarks, and plotter
- tests/ - Shared patient datasets (test_patients.py), uncertainty tests (test_uncertainty.py), and control surface plotter (test_control_surfaces.py)