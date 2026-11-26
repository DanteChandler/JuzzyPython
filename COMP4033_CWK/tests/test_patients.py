"""
Shared test patient datasets for Case 1 (numeric midpoints) and Case 2 (intervals).
Add additional test scenarios or test scripts inside this tests/ folder.
"""

# Case 1: Numeric Test Cases (Includes boundary/transition patient to highlight operator & defuzz differences)
TEST_PATIENTS_NUMERIC = [
    ("Patient 1 (Healthy Adult, Mild Headache)",       36.8, 2.0, 28.0),
    ("Patient 2 (Child 8yo, Moderate Fever & Pain)",   38.6, 5.0,  8.0),
    ("Patient 3 (Infant 2yo, High Fever & Distress)",  39.2, 6.0,  2.0),
    ("Patient 4 (Older Adult 68yo, Severe Headache)",  37.0, 8.5, 68.0),
    ("Patient 5 (Adult, Hyperpyrexia Crisis)",         41.2, 7.0, 35.0),
    ("Patient 6 (Elderly, Hypothermia Exposure)",      34.0, 1.0, 78.0),
    ("Patient 7 (Borderline Fever 37.8C, Mod Pain)",   37.8, 6.5, 55.0),
]

# Case 2: Interval Test Cases
# Note: Each interval is centered on the exact same midpoint as the Case 1 test patients
TEST_PATIENTS_INTERVALS = [
    ("Patient 1 (Healthy Adult, Mild Headache)",       (36.4, 37.2), (1.0, 3.0), (26.0, 30.0)),
    ("Patient 2 (Child ~8yo, Mod Fever & Pain)",       (38.1, 39.1), (4.0, 6.0), ( 7.0,  9.0)),
    ("Patient 3 (Infant ~2yo, High Fever & Distress)", (38.7, 39.7), (5.0, 7.0), ( 1.5,  2.5)),
    ("Patient 4 (Older Adult ~68yo, Severe Headache)", (36.5, 37.5), (7.5, 9.5), (65.0, 71.0)),
    ("Patient 5 (Adult, Hyperpyrexia Crisis)",         (40.7, 41.7), (6.0, 8.0), (32.0, 38.0)),
    ("Patient 6 (Elderly, Hypothermia Exposure)",      (33.5, 34.5), (0.5, 1.5), (75.0, 81.0)),
]