"""
Case 1 Comparative Model: Type-1 Mamdani (All inputs are numeric)

Note: The FINAL model for Case 1 is Zeroth-Order TSK (case1/tsk_model.py). This is simply used for comparison
"""

import math
from rules import setup_juzzy_path, get_rules

setup_juzzy_path()

from juzzyPython.generic.Tuple import Tuple
from juzzyPython.generic.Input import Input
from juzzyPython.generic.Output import Output
from juzzyPython.type1.sets.T1MF_Triangular import T1MF_Triangular
from juzzyPython.type1.sets.T1MF_Trapezoidal import T1MF_Trapezoidal
from juzzyPython.type1.sets.T1MF_Gaussian import T1MF_Gaussian
from juzzyPython.type1.system.T1_Antecedent import T1_Antecedent
from juzzyPython.type1.system.T1_Consequent import T1_Consequent
from juzzyPython.type1.system.T1_Rule import T1_Rule
from juzzyPython.type1.system.T1_Rulebase import T1_Rulebase


class Case1FuzzyLogicSystem:
    def __init__(self, mf_style="piecewise", operator_family="min_max", rule_mode="compact_21"):
        """
        Initialises the Case 1 Mamdani Fuzzy Logic System with configurable options:
        - mf_style: "piecewise", "gaussian", or "hybrid" for membership function shapes
        - operator_family: "min_max" for Minimum/Maximum operators, "product" for Product/Probabilistic-Sum operators
        - rule_mode: "compact_21" (21 rules) or "exhaustive_64" (full 64-rule grid)
        """
        self.mf_style = mf_style
        self.operator_family = operator_family
        self.rule_mode = rule_mode

        # Defining inputs and outputs with their ranges
        self.temp_in = Input("Temperature (°C)", Tuple(0.0, 50.0))
        self.headache_in = Input("Headache Severity", Tuple(0.0, 10.0))
        self.age_in = Input("Patient Age (Years)", Tuple(0.0, 130.0))
        self.urgency_out = Output("Urgency Score", Tuple(0.0, 100.0))
        self.urgency_out.setDiscretisationLevel(200)

        # Defining membership functions for each input and output variable
        # Option 1: Piecewise (Trapezoidal at domain boundaries, Triangular for middle)
        if mf_style == "piecewise":
            # Temperature all defined by NHS and NICE
            # Hypothermia (<35°C)
            # Normal (36.0–37.5°C)
            # Moderate Fever (38.0–40.0°C)
            # Hyperpyrexia (>40.0°C)
            self.temp_hypo = T1MF_Trapezoidal("Hypothermia", [0.0, 0.0, 35.0, 36.0])
            self.temp_norm = T1MF_Triangular("Normal", 35.2, 36.8, 38.0)
            self.temp_fever = T1MF_Triangular("Moderate Fever", 37.5, 38.8, 40.0)
            self.temp_hyper = T1MF_Trapezoidal("Hyperpyrexia", [39.0, 40.0, 50.0, 50.0])

            # Headache
            # This is all based off of the NHS Headache Severity Scale (0-10)
            self.head_none = T1MF_Trapezoidal("No Pain", [0.0, 0.0, 0.5, 1.5])
            self.head_mild = T1MF_Triangular("Mild Pain", 0.5, 2.0, 4.0)
            self.head_mod = T1MF_Triangular("Moderate Pain", 3.0, 5.0, 7.0)
            self.head_sev = T1MF_Trapezoidal("Severe Pain", [6.0, 7.5, 10.0, 10.0])

            # Age
            # Under 5 (<5): Considered vulnerable/high risk by NHS
            # Child (5–12): Still need to be cautious according to NICE
            # Adult (13–50): General healthy adult population, less vulnerable to complications
            # Older Adult (>50): Considered more vulnerable/high risk again by NHS
            self.age_u5 = T1MF_Trapezoidal("Under 5", [0.0, 0.0, 5.0, 6.0])
            self.age_child = T1MF_Triangular("Child (5-12)", 5.0, 8.5, 13.0)
            self.age_adult = T1MF_Trapezoidal("Adult", [12.0, 18.0, 50.0, 65.0])
            self.age_older = T1MF_Trapezoidal("Older Adult (>50)", [50.0, 65.0, 130.0, 130.0])

            # Urgency (output)
            # Low: Home care or GP checkup, maybe even nothing
            # Medium: Monitor and follow up with GP or hospital
            # High: Immediate hospital referral or emergency services
            self.urg_low = T1MF_Trapezoidal("Low", [0.0, 0.0, 20.0, 40.0])
            self.urg_med = T1MF_Triangular("Medium", 25.0, 50.0, 75.0)
            self.urg_high = T1MF_Trapezoidal("High", [60.0, 80.0, 100.0, 100.0])

        # Option 2: Gaussian (Smooth bell curves with infinite support, modelling biological smoothness)
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

            self.urg_low = T1MF_Gaussian("Low", 15.0, 12.0)
            self.urg_med = T1MF_Gaussian("Medium", 50.0, 12.0)
            self.urg_high = T1MF_Gaussian("High", 85.0, 12.0)

        # Option 3: Hybrid (Trapezoidal at domain extremes + Gaussian interior)
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

            self.urg_low = T1MF_Trapezoidal("Low", [0.0, 0.0, 20.0, 40.0])
            self.urg_med = T1MF_Gaussian("Medium", 50.0, 12.0)
            self.urg_high = T1MF_Trapezoidal("High", [60.0, 80.0, 100.0, 100.0])

        # Creating antecedents and consequents
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

        self.con_urg = {
            "Low": T1_Consequent(self.urg_low, self.urgency_out, "Low"),
            "Medium": T1_Consequent(self.urg_med, self.urgency_out, "Medium"),
            "High": T1_Consequent(self.urg_high, self.urgency_out, "High"),
        }

        # Building rule base
        # JuzzyPy Operator codes: 0 = Minimum (t-norm) & Maximum (implication), 1 = Product (t-norm) & Product (implication)
        self.rulebase = T1_Rulebase()
        if operator_family == "product":
            self.rulebase.setImplicationMethod(1) # Product implication
            self.rulebase.setInferenceMethod(1)   # Product t-norm
        else:
            self.rulebase.setImplicationMethod(0) # Minimum implication
            self.rulebase.setInferenceMethod(0)   # Minimum t-norm

        self._build_rules()

    def _build_rules(self):
        """Builds either the 21 compact rules or the 64 exhaustive rules."""
        self.rules_list = get_rules(
            self.rule_mode,
            self.ant_temp.keys(),
            self.ant_head.keys(),
            self.ant_age.keys(),
        )
        self.rules_log = []
        for i, (t_label, h_label, a_label, urg_key) in enumerate(self.rules_list, start=1):
            antecedents = []
            parts = []
            if t_label is not None:
                antecedents.append(self.ant_temp[t_label])
                parts.append(f"Temp is {t_label}")
            if h_label is not None:
                antecedents.append(self.ant_head[h_label])
                parts.append(f"Headache is {h_label}")
            if a_label is not None:
                antecedents.append(self.ant_age[a_label])
                parts.append(f"Age is {a_label}")

            rule = T1_Rule(antecedents, consequent=self.con_urg[urg_key])
            self.rulebase.addRule(rule)
            self.rules_log.append(f"R{i:02d}: IF " + " AND ".join(parts) + f" THEN Urgency is {urg_key}")

    def evaluate_patient(self, temp_val, headache_val, age_val, defuzz_type=1):
        """
        Evaluates a patient and computes the urgency score using the specified defuzzification method:
        - defuzz_type: 0 = Height, 1 = Centroid (default), 2 = Mean of Maxima (MoM), 3 = Bisector 
        """
        self.temp_in.setInput(float(temp_val))
        self.headache_in.setInput(float(headache_val))
        self.age_in.setInput(float(age_val))

        output_dict = self.rulebase.evaluate(defuzz_type)
        return output_dict[self.urgency_out]

    def evaluate_all_defuzz_methods(self, temp_val, headache_val, age_val, steps=200):
        """
        Evaluates a patient and computes defuzzification techniques for the urgency score:
        1. Centroid (JuzzyPy native evaluate(1))
        2. Height (JuzzyPy native evaluate(0))
        3. Mean of Maxima (MoM)
        4. Bisector
        """
        centroid_score = self.evaluate_patient(temp_val, headache_val, age_val, defuzz_type=1)
        height_score = self.evaluate_patient(temp_val, headache_val, age_val, defuzz_type=0)

        x_vals = [i * (100.0 / steps) for i in range(steps + 1)]
        agg_mu = [0.0] * len(x_vals)

        for t_label, h_label, a_label, urg_key in self.rules_list:
            memberships = []
            if t_label is not None:
                memberships.append(self.ant_temp[t_label].getFS())
            if h_label is not None:
                memberships.append(self.ant_head[h_label].getFS())
            if a_label is not None:
                memberships.append(self.ant_age[a_label].getFS())

            firing = math.prod(memberships) if self.operator_family == "product" else min(memberships)

            if firing > 0.0:
                con_mf = self.con_urg[urg_key].getMF()
                for idx, x in enumerate(x_vals):
                    implied_val = firing * con_mf.getFS(x) if self.operator_family == "product" else min(firing, con_mf.getFS(x))
                    if self.operator_family == "product":
                        agg_mu[idx] = agg_mu[idx] + implied_val - (agg_mu[idx] * implied_val)
                    else:
                        agg_mu[idx] = max(agg_mu[idx], implied_val)

        # Mean of Maxima (MoM)
        max_mu = max(agg_mu) # Finds the maximum membership value
        if max_mu > 0:
            max_x = [x_vals[i] for i, val in enumerate(agg_mu) if abs(val - max_mu) < 1e-6]
            mom_score = sum(max_x) / len(max_x)
        else:
            mom_score = 0.0

        # Bisector
        total_area = sum(agg_mu) # Finds the total area under the aggregated membership function
        running_area = 0.0
        bisector_score = 0.0
        for i, val in enumerate(agg_mu):
            running_area += val
            if running_area >= total_area / 2.0:
                bisector_score = x_vals[i]
                break

        return {
            "Centroid": centroid_score,
            "Height": height_score,
            "MoM": mom_score,
            "Bisector": bisector_score,
        }