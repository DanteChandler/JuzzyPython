"""
Rules for the Fuzzy Inference System
"""

import sys
import itertools
from pathlib import Path


def setup_juzzy_path():
    """Adds the JuzzyPython folder to sys.path so imports work from any subfolder."""
    juzzy_path = str(Path(__file__).resolve().parent / "JuzzyPython")
    if juzzy_path not in sys.path:
        sys.path.insert(0, juzzy_path)


# Final 21-Rule Base
COMPACT_21_RULES = [
    # A severe symptom such as Hypothermia, Hyperpyrexia or Severe Pain automatically makes the consequent high urgency
    ("Hypothermia", None, None, "High"),
    ("Hyperpyrexia", None, None, "High"),
    (None, "Severe Pain", None, "High"),

    ("Normal", "No Pain", None, "Low"),
    ("Normal", "Mild Pain", None, "Low"),

    # One moderate symptom makes the consequent low for adults but medium for Under 5s, children and Older Adults
    ("Moderate Fever", "No Pain", "Under 5", "Medium"),
    ("Moderate Fever", "No Pain", "Child (5-12)", "Medium"),
    ("Moderate Fever", "No Pain", "Adult", "Low"),
    ("Moderate Fever", "No Pain", "Older Adult (>50)", "Medium"),

    ("Moderate Fever", "Mild Pain", "Under 5", "Medium"),
    ("Moderate Fever", "Mild Pain", "Child (5-12)", "Medium"),
    ("Moderate Fever", "Mild Pain", "Adult", "Low"),
    ("Moderate Fever", "Mild Pain", "Older Adult (>50)", "Medium"),

    ("Normal", "Moderate Pain", "Under 5", "Medium"),
    ("Normal", "Moderate Pain", "Child (5-12)", "Medium"),
    ("Normal", "Moderate Pain", "Adult", "Low"),
    ("Normal", "Moderate Pain", "Older Adult (>50)", "Medium"),

    # Two moderate symptoms makes the consequent High for under 5s, medium for children, medium for adults and high for older Adults
    ("Moderate Fever", "Moderate Pain", "Under 5", "High"),
    ("Moderate Fever", "Moderate Pain", "Child (5-12)", "Medium"),
    ("Moderate Fever", "Moderate Pain", "Adult", "Medium"),
    ("Moderate Fever", "Moderate Pain", "Older Adult (>50)", "High"),
]


def determine_consequent(t_label, h_label, a_label):
    """
    Assigns an Urgency level (Low, Medium, High) to each of the 64 rules using 3 main concepts:

    1. Extreme temperatures are prioritised (32 rules -> High):
       - Hypothermia (<35°C) or Hyperpyrexia (>40°C) is always 'High', regardless of age or pain

    2. Severe Headaches (7-10 -> High or Medium):
       - AND Moderate Fever: 'High' for all ages (concern for serious infection like meningitis)
       - AND Normal Temp: 'High' for high-risk ages (Under 5, Older Adult), 'Medium' for Child/Adult

    3. Everything Else Depends on Age & Pain:
       - Moderate Fever (38-40°C): 'High' for Under 5s with moderate pain; 'Low' for Adults with
         no/mild pain, 'Medium' for everyone else
       - Normal Temp (36-37.5°C): 'Low' for all ages if AND no/mild pain. AND moderate pain, 'Low' for
         Adults and 'Medium' for Under 5, Child, and Older Adult as a precaution
    """
    # Rule Concept 1: Extreme Temperatures -> High Urgency
    if t_label in ("Hypothermia", "Hyperpyrexia"):
        return "High"

    # Rule Concept 2: Moderate Fever
    if t_label == "Moderate Fever":
        if h_label == "Severe Pain":
            return "High"  # Fever + severe headache = red flag
        if a_label == "Under 5" and h_label == "Moderate Pain":
            return "High"  # Febrile infant/young child in distress
        if a_label == "Adult" and h_label in ("No Pain", "Mild Pain"):
            return "Low"   # Resilient adult with mild viral fever
        return "Medium"    # All other moderate fever cases warrant clinical monitoring

    # Rule Concept 3: Normal Temperature
    if h_label == "Severe Pain":
        if a_label in ("Under 5", "Older Adult (>50)"):
            return "High"  # Unexplained severe headache in high-risk age brackets
        return "Medium"    # Severe headache in Child/Adult without fever

    if h_label == "Moderate Pain":
        if a_label == "Adult":
            return "Low"
        return "Medium"    # Cautious monitoring for Under 5, Child, and Older Adult

    # Normal Temp + No Pain / Mild Pain
    return "Low"


def get_rules(rule_mode="compact_21", temp_keys=None, head_keys=None, age_keys=None):
    """
    Returns the rule list as (temp_label, head_label, age_label, urgency_label) tuples.
    - "compact_21": Final 21-rule set
    - "exhaustive_64": Full 4 x 4 x 4 = 64-rule grid for comparison
    """
    if rule_mode == "exhaustive_64":
        return [
            (t, h, a, determine_consequent(t, h, a))
            for t, h, a in itertools.product(temp_keys, head_keys, age_keys)
        ]
    return list(COMPACT_21_RULES)