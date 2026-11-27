"""
COMP4033 Coursework - Case 1 FINAL MODEL: Zeroth-Order TSK (All inputs are numeric)

Summary:
- System Type: Type-1 Zeroth-Order Takagi-Sugeno-Kang (TSK) Fuzzy Inference System
- 3 Inputs: Temperature [0, 50], Headache [0, 10], Age [0, 130]
- Output Constants: Low = 10.0, Medium = 50.0, High = 90.0
- MF Shapes: Configurable between Piecewise (Final), Gaussian, and Hybrid
- Rule Base: Configurable between Compact 21-rule set (Final) and Exhaustive 64-rule grid
- Operators: Configurable between Minimum (Final) and Product t-norm
"""

import math
from rules import setup_juzzy_path, get_rules

setup_juzzy_path()

from juzzyPython.generic.Tuple import Tuple
from juzzyPython.generic.Input import Input
from juzzyPython.type1.sets.T1MF_Triangular import T1MF_Triangular
from juzzyPython.type1.sets.T1MF_Trapezoidal import T1MF_Trapezoidal
from juzzyPython.type1.sets.T1MF_Gaussian import T1MF_Gaussian
from juzzyPython.type1.system.T1_Antecedent import T1_Antecedent


class Case1TSKFuzzyLogicSystem:
    TSK_CONSTANTS = {"Low": 10.0, "Medium": 50.0, "High": 100.0}

    def __init__(self, mf_style="piecewise", operator_family="min_max", rule_mode="compact_21"):
        """
        Initialises the Final Case 1 Zeroth-Order TSK System with configurable options:
        - mf_style: "piecewise" (final), "gaussian", or "hybrid"
        - operator_family: "min_max" (final) or "product"
        - rule_mode: "compact_21" (final) or "exhaustive_64"
        """
        self.mf_style = mf_style
        self.operator_family = operator_family
        self.rule_mode = rule_mode

        # Defining inputs with their ranges
        self.temp_in = Input("Temperature (°C)", Tuple(0.0, 50.0))
        self.headache_in = Input("Headache Severity", Tuple(0.0, 10.0))
        self.age_in = Input("Patient Age (Years)", Tuple(0.0, 130.0))

        # Option 1: Piecewise (Trapezoidal at domain boundaries, Triangular for middle)
        if mf_style == "piecewise":
            # Temperature all defined by NHS and NICE
            # Hypothermia (<35°C), Normal (36.0–37.5°C), Moderate Fever (38.0–40.0°C), Hyperpyrexia (>40.0°C)
            self.temp_hypo = T1MF_Trapezoidal("Hypothermia", [0.0, 0.0, 35.0, 36.0])
            self.temp_norm = T1MF_Triangular("Normal", 35.2, 36.8, 38.0)
            self.temp_fever = T1MF_Triangular("Moderate Fever", 37.5, 38.8, 40.0)
            self.temp_hyper = T1MF_Trapezoidal("Hyperpyrexia", [39.0, 40.0, 50.0, 50.0])

            # Headache (NHS Headache Severity Scale 0-10)
            self.head_none = T1MF_Trapezoidal("No Pain", [0.0, 0.0, 0.5, 1.5])
            self.head_mild = T1MF_Triangular("Mild Pain", 0.5, 2.0, 4.0)
            self.head_mod = T1MF_Triangular("Moderate Pain", 3.0, 5.0, 7.0)
            self.head_sev = T1MF_Trapezoidal("Severe Pain", [6.0, 7.5, 10.0, 10.0])

            # Age (standardised 50-65 transition for Older Adult)
            self.age_u5 = T1MF_Trapezoidal("Under 5", [0.0, 0.0, 5.0, 6.0])
            self.age_child = T1MF_Triangular("Child (5-12)", 5.0, 8.5, 13.0)
            self.age_adult = T1MF_Trapezoidal("Adult", [12.0, 18.0, 50.0, 65.0])
            self.age_older = T1MF_Trapezoidal("Older Adult (>50)", [50.0, 65.0, 130.0, 130.0])

        # Option 2: Gaussian (Smooth bell curves with infinite support)
        elif mf_style == "gaussian":
            self.temp_hypo = T1MF_Gaussian("Hypothermia", 32.0, 2.5)
            self.temp_norm = T1MF_Gaussian("Normal", 36.8, 0.6)
            self.temp_fever = T1MF_Gaussian("Moderate Fever", 38.8, 0.6)
            self.temp_hyper = T1MF_Gaussian("Hyperpyrexia", 42.0, 1.5)

            self.head_none = T1MF_Gaussian("No Pain", 0.0, 0.6)
            self.head_mild = T1MF_Gaussian("Mild Pain", 2.0, 0.8)
            self.head_mod = T1MF_Gaussian("Moderate Pain", 5.0, 0.9)
            self.head_sev = T1MF_Gaussian("Severe Pain", 9.0, 1.2)

            self.age_u5 = T1MF_Gaussian("Under 5", 2.0, 2.0)
            self.age_child = T1MF_Gaussian("Child (5-12)", 8.5, 2.0)
            self.age_adult = T1MF_Gaussian("Adult", 32.0, 12.0)
            self.age_older = T1MF_Gaussian("Older Adult (>50)", 80.0, 15.0)

        # Option 3: Hybrid (Trapezoidal at domain extremes + Gaussian interior sets)
        elif mf_style == "hybrid":
            self.temp_hypo = T1MF_Trapezoidal("Hypothermia", [0.0, 0.0, 35.0, 36.0])
            self.temp_norm = T1MF_Gaussian("Normal", 36.8, 0.6)
            self.temp_fever = T1MF_Gaussian("Moderate Fever", 38.8, 0.6)
            self.temp_hyper = T1MF_Trapezoidal("Hyperpyrexia", [39.0, 40.0, 50.0, 50.0])

            self.head_none = T1MF_Trapezoidal("No Pain", [0.0, 0.0, 0.5, 1.5])
            self.head_mild = T1MF_Gaussian("Mild Pain", 2.0, 0.8)
            self.head_mod = T1MF_Gaussian("Moderate Pain", 5.0, 0.9)
            self.head_sev = T1MF_Trapezoidal("Severe Pain", [6.0, 7.5, 10.0, 10.0])

            self.age_u5 = T1MF_Trapezoidal("Under 5", [0.0, 0.0, 5.0, 6.0])
            self.age_child = T1MF_Gaussian("Child (5-12)", 8.5, 2.0)
            self.age_adult = T1MF_Trapezoidal("Adult", [12.0, 18.0, 50.0, 65.0])
            self.age_older = T1MF_Trapezoidal("Older Adult (>50)", [50.0, 65.0, 130.0, 130.0])

        # Creating antecedents
        self.ant_temp = {
            "Hypothermia": T1_Antecedent(self.temp_hypo, self.temp_in, "Hypothermia"),
            "Normal": T1_Antecedent(self.temp_norm, self.temp_in, "Normal"),
            "Moderate Fever": T1_Antecedent(self.temp_fever, self.temp_in, "Moderate Fever"),
            "Hyperpyrexia": T1_Antecedent(self.temp_hyper, self.temp_in, "Hyperpyrexia"),
        }

        self.ant_head = {
            "No Pain": T1_Antecedent(self.head_none, self.headache_in, "No Pain"),
            "Mild Pain": T1_Antecedent(self.head_mild, self.headache_in, "Mild Pain"),
            "Moderate Pain": T1_Antecedent(self.head_mod, self.headache_in, "Moderate Pain"),
            "Severe Pain": T1_Antecedent(self.head_sev, self.headache_in, "Severe Pain"),
        }

        self.ant_age = {
            "Under 5": T1_Antecedent(self.age_u5, self.age_in, "Under 5"),
            "Child (5-12)": T1_Antecedent(self.age_child, self.age_in, "Child (5-12)"),
            "Adult": T1_Antecedent(self.age_adult, self.age_in, "Adult"),
            "Older Adult (>50)": T1_Antecedent(self.age_older, self.age_in, "Older Adult (>50)"),
        }

        self._build_rules()

    def _build_rules(self):
        self.rules_list = get_rules(
            self.rule_mode,
            self.ant_temp.keys(),
            self.ant_head.keys(),
            self.ant_age.keys(),
        )
        self.rules_log = []
        for i, (t, h, a, urg) in enumerate(self.rules_list, start=1):
            parts = []
            if t is not None:
                parts.append(f"Temp is {t}")
            if h is not None:
                parts.append(f"Headache is {h}")
            if a is not None:
                parts.append(f"Age is {a}")
            self.rules_log.append(
                f"R{i:02d}: IF " + " AND ".join(parts) + f" THEN Urgency = {self.TSK_CONSTANTS[urg]:.0f} ({urg})"
            )

    def evaluate_patient(self, temp_val, headache_val, age_val):
        """Zeroth-order TSK weighted-average output."""
        self.temp_in.setInput(float(temp_val))
        self.headache_in.setInput(float(headache_val))
        self.age_in.setInput(float(age_val))

        top = 0.0
        bottom = 0.0
        for t, h, a, urg in self.rules_list:
            memberships = []
            if t is not None:
                memberships.append(self.ant_temp[t].getFS())
            if h is not None:
                memberships.append(self.ant_head[h].getFS())
            if a is not None:
                memberships.append(self.ant_age[a].getFS())

            if self.operator_family == "product":
                firing = math.prod(memberships)
            else:
                firing = min(memberships)

            top += firing * self.TSK_CONSTANTS[urg]
            bottom += firing

        if bottom == 0.0:
            return 0.0
        return top / bottom