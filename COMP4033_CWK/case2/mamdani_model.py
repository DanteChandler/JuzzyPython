"""
Case 2 FINAL MODEL: Interval Type-2 Mamdani (All inputs are interval-valued)

Summary:
- System Type: Interval Type-2 Mamdani Fuzzy Logic System
- 3 Inputs: Temperature [0, 50], Headache [0, 10], Age [0, 130]
- 1 Output: Urgency [0, 100] (3 terms: Low, Medium, High)
- MF Shapes: Interval Type-2 Trapezoidal at domain boundaries/plateaus, IT2 Triangular for middle (with UMF and LMF)
- Rule Base: Configurable between Compact 21-rule set (Final) and Exhaustive 64-rule grid
- Operators: Min/Max t-norm and s-norm
- Defuzzification:
    - Numeric: Centroid of Set (COS) / Centroid of Area (COA)
    - Linguistic: Jaccard Similarity (Low, Medium, High)
"""

import itertools
from rules import setup_juzzy_path, get_rules

setup_juzzy_path()

from juzzyPython.generic.Tuple import Tuple
from juzzyPython.generic.Input import Input
from juzzyPython.generic.Output import Output
from juzzyPython.type1.sets.T1MF_Triangular import T1MF_Triangular
from juzzyPython.type1.sets.T1MF_Trapezoidal import T1MF_Trapezoidal
from juzzyPython.intervalType2.sets.IntervalT2MF_Triangular import IntervalT2MF_Triangular as IT2MF_Triangular
from juzzyPython.intervalType2.sets.IntervalT2MF_Trapezoidal import IntervalT2MF_Trapezoidal as IT2MF_Trapezoidal
from juzzyPython.intervalType2.sets.IntervalT2MF_Cylinder import IntervalT2MF_Cylinder
from juzzyPython.intervalType2.sets.IntervalT2MF_Intersection import IntervalT2MF_Intersection
from juzzyPython.intervalType2.sets.IntervalT2MF_Union import IntervalT2MF_Union
from juzzyPython.intervalType2.system.IT2_Antecedent import IT2_Antecedent
from juzzyPython.intervalType2.system.IT2_Consequent import IT2_Consequent
from juzzyPython.intervalType2.system.IT2_Rule import IT2_Rule
from juzzyPython.intervalType2.system.IT2_Rulebase import IT2_Rulebase
from juzzyPython.similarity.JaccardSimilarity import JaccardSimilarity


class Case2FuzzyLogicSystem:
    def __init__(self, rule_mode="compact_21"):
        """
        Initialises the Final Case 2 Interval Type-2 Mamdani System:
        - rule_mode: "compact_21" (final) or "exhaustive_64" (for comparison)
        """
        self.rule_mode = rule_mode

        # Defining inputs and outputs with their ranges
        self.temp_in = Input("Temperature (°C)", Tuple(0, 50))
        self.headache_in = Input("Headache Severity", Tuple(0, 10))
        self.age_in = Input("Patient Age (Years)", Tuple(0, 130))
        self.urgency_out = Output("Urgency Score", Tuple(0.0, 100.0))
        self.urgency_out.setDiscretisationLevel(200)

        # Defining Interval Type-2 membership functions (requires an Upper MF and a Lower MF to form the FOU)

        # Temperature
        self.temp_hypo_umf = T1MF_Trapezoidal("Hypothermia_UMF", [0.0, 0.0, 35.0, 36.2])
        self.temp_hypo_lmf = T1MF_Trapezoidal("Hypothermia_LMF", [0.0, 0.0, 34.5, 35.5])
        self.temp_hypo = IT2MF_Trapezoidal("Hypothermia", self.temp_hypo_umf, self.temp_hypo_lmf)

        self.temp_norm_umf = T1MF_Triangular("Normal_UMF", 35.0, 36.8, 38.2)
        self.temp_norm_lmf = T1MF_Triangular("Normal_LMF", 35.5, 36.8, 37.7)
        self.temp_norm = IT2MF_Triangular("Normal", self.temp_norm_umf, self.temp_norm_lmf)

        self.temp_fever_umf = T1MF_Triangular("Moderate Fever_UMF", 37.3, 38.8, 40.2)
        self.temp_fever_lmf = T1MF_Triangular("Moderate Fever_LMF", 37.8, 38.8, 39.7)
        self.temp_fever = IT2MF_Triangular("Moderate Fever", self.temp_fever_umf, self.temp_fever_lmf)

        self.temp_hyper_umf = T1MF_Trapezoidal("Hyperpyrexia_UMF", [38.8, 40.0, 50.0, 50.0])
        self.temp_hyper_lmf = T1MF_Trapezoidal("Hyperpyrexia_LMF", [39.3, 40.3, 50.0, 50.0])
        self.temp_hyper = IT2MF_Trapezoidal("Hyperpyrexia", self.temp_hyper_umf, self.temp_hyper_lmf)

        # Headache [0 - 10]
        self.head_none_umf = T1MF_Trapezoidal("No Pain_UMF", [0.0, 0.0, 0.7, 1.8])
        self.head_none_lmf = T1MF_Trapezoidal("No Pain_LMF", [0.0, 0.0, 0.3, 1.2])
        self.head_none = IT2MF_Trapezoidal("No Pain", self.head_none_umf, self.head_none_lmf)

        self.head_mild_umf = T1MF_Triangular("Mild Pain_UMF", 0.3, 2.0, 4.3)
        self.head_mild_lmf = T1MF_Triangular("Mild Pain_LMF", 0.8, 2.0, 3.7)
        self.head_mild = IT2MF_Triangular("Mild Pain", self.head_mild_umf, self.head_mild_lmf)

        self.head_mod_umf = T1MF_Triangular("Moderate Pain_UMF", 2.7, 5.0, 7.3)
        self.head_mod_lmf = T1MF_Triangular("Moderate Pain_LMF", 3.3, 5.0, 6.7)
        self.head_mod = IT2MF_Triangular("Moderate Pain", self.head_mod_umf, self.head_mod_lmf)

        self.head_sev_umf = T1MF_Trapezoidal("Severe Pain_UMF", [5.7, 7.2, 10.0, 10.0])
        self.head_sev_lmf = T1MF_Trapezoidal("Severe Pain_LMF", [6.3, 7.8, 10.0, 10.0])
        self.head_sev = IT2MF_Trapezoidal("Severe Pain", self.head_sev_umf, self.head_sev_lmf)

        # Age [0 - 130 Years] (aligned to 50-65 transition to match Case 1)
        self.age_u5_umf = T1MF_Trapezoidal("Under 5_UMF", [0.0, 0.0, 5.0, 6.5])
        self.age_u5_lmf = T1MF_Trapezoidal("Under 5_LMF", [0.0, 0.0, 4.5, 5.5])
        self.age_u5 = IT2MF_Trapezoidal("Under 5", self.age_u5_umf, self.age_u5_lmf)

        self.age_child_umf = T1MF_Triangular("Child (5-12)_UMF", 4.5, 8.5, 13.5)
        self.age_child_lmf = T1MF_Triangular("Child (5-12)_LMF", 5.5, 8.5, 12.5)
        self.age_child = IT2MF_Triangular("Child (5-12)", self.age_child_umf, self.age_child_lmf)

        self.age_adult_umf = T1MF_Trapezoidal("Adult_UMF", [11.5, 17.5, 51.0, 66.0])
        self.age_adult_lmf = T1MF_Trapezoidal("Adult_LMF", [12.5, 18.5, 49.0, 64.0])
        self.age_adult = IT2MF_Trapezoidal("Adult", self.age_adult_umf, self.age_adult_lmf)

        self.age_older_umf = T1MF_Trapezoidal("Older Adult (>50)_UMF", [49.0, 64.0, 130.0, 130.0])
        self.age_older_lmf = T1MF_Trapezoidal("Older Adult (>50)_LMF", [51.0, 66.0, 130.0, 130.0])
        self.age_older = IT2MF_Trapezoidal("Older Adult (>50)", self.age_older_umf, self.age_older_lmf)

        # Urgency (output) [0 - 100]
        self.urg_low_umf = T1MF_Trapezoidal("Low_UMF", [0.0, 0.0, 22.0, 42.0])
        self.urg_low_lmf = T1MF_Trapezoidal("Low_LMF", [0.0, 0.0, 18.0, 38.0])
        self.urg_low = IT2MF_Trapezoidal("Low", self.urg_low_umf, self.urg_low_lmf)

        self.urg_med_umf = T1MF_Triangular("Medium_UMF", 23.0, 50.0, 77.0)
        self.urg_med_lmf = T1MF_Triangular("Medium_LMF", 27.0, 50.0, 73.0)
        self.urg_med = IT2MF_Triangular("Medium", self.urg_med_umf, self.urg_med_lmf)

        self.urg_high_umf = T1MF_Trapezoidal("High_UMF", [58.0, 78.0, 100.0, 100.0])
        self.urg_high_lmf = T1MF_Trapezoidal("High_LMF", [62.0, 82.0, 100.0, 100.0])
        self.urg_high = IT2MF_Trapezoidal("High", self.urg_high_umf, self.urg_high_lmf)

        # Creating IT2 antecedents and consequents
        self.ant_temp = {
            "Hypothermia": IT2_Antecedent(self.temp_hypo, self.temp_in, "Hypothermia"),
            "Normal": IT2_Antecedent(self.temp_norm, self.temp_in, "Normal"),
            "Moderate Fever": IT2_Antecedent(self.temp_fever, self.temp_in, "Moderate Fever"),
            "Hyperpyrexia": IT2_Antecedent(self.temp_hyper, self.temp_in, "Hyperpyrexia"),
        }

        self.ant_head = {
            "No Pain": IT2_Antecedent(self.head_none, self.headache_in, "No Pain"),
            "Mild Pain": IT2_Antecedent(self.head_mild, self.headache_in, "Mild Pain"),
            "Moderate Pain": IT2_Antecedent(self.head_mod, self.headache_in, "Moderate Pain"),
            "Severe Pain": IT2_Antecedent(self.head_sev, self.headache_in, "Severe Pain"),
        }

        self.ant_age = {
            "Under 5": IT2_Antecedent(self.age_u5, self.age_in, "Under 5"),
            "Child (5-12)": IT2_Antecedent(self.age_child, self.age_in, "Child (5-12)"),
            "Adult": IT2_Antecedent(self.age_adult, self.age_in, "Adult"),
            "Older Adult (>50)": IT2_Antecedent(self.age_older, self.age_in, "Older Adult (>50)"),
        }

        self.con_urg = {
            "Low": IT2_Consequent(self.urg_low, self.urgency_out, "Low"),
            "Medium": IT2_Consequent(self.urg_med, self.urgency_out, "Medium"),
            "High": IT2_Consequent(self.urg_high, self.urgency_out, "High"),
        }

        # Building rule base
        self.rulebase = IT2_Rulebase()
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

            rule = IT2_Rule(antecedents, consequent=self.con_urg[urg_key])
            self.rulebase.addRule(rule)
            self.rules_log.append(f"R{i:02d}: IF " + " AND ".join(parts) + f" THEN Urgency is {urg_key}")

    def _create_interval_input_mf(self, name, interval_tuple, domain_tuple):
        """
        Creates an Interval Type-2 triangular membership function for an input variable
        based on the provided interval and the variable's domain.
        """
        low, high = float(interval_tuple[0]), float(interval_tuple[1])
        d_min, d_max = float(domain_tuple.getLeft()), float(domain_tuple.getRight())

        a = max(d_min, low)
        c = min(d_max, high)
        
        # This is to stop the triangle from collapsing into a line if the interval is too narrow
        if abs(c - a) < 1e-6:
            a = max(d_min, a - 1e-4)
            c = min(d_max, c + 1e-4)
        b = (a + c) / 2.0

        return T1MF_Triangular(f"{name}_IntervalInput", a, b, c)

    def evaluate_patient_interval(self, temp_interval, headache_interval, age_interval, type_reduction=0):
        """
        Evaluates a patient with interval-valued inputs and returns the defuzzified urgency score
        along with the minimum and maximum urgency values from the type-reduced output set.
        """
        t_pts = [float(temp_interval[0]), (temp_interval[0] + temp_interval[1]) / 2.0, float(temp_interval[1])]
        h_pts = [float(headache_interval[0]), (headache_interval[0] + headache_interval[1]) / 2.0, float(headache_interval[1])]
        a_pts = [float(age_interval[0]), (age_interval[0] + age_interval[1]) / 2.0, float(age_interval[1])]

        left_bounds = []
        right_bounds = []

        for t_val, h_val, a_val in itertools.product(t_pts, h_pts, a_pts):
            self.temp_in.setInput(t_val)
            self.headache_in.setInput(h_val)
            self.age_in.setInput(a_val)

            try:
                centroid_dict = self.rulebase.evaluateGetCentroid(type_reduction)
                tr_tuple = centroid_dict[self.urgency_out][0]
                left_bounds.append(float(tr_tuple.getLeft()))
                right_bounds.append(float(tr_tuple.getRight()))
            except ZeroDivisionError:
                continue

        u_min = min(left_bounds)
        u_max = max(right_bounds)
        defuzz_score = (u_min + u_max) / 2.0
        return defuzz_score, (u_min, u_max)

    def linguistic_output(self, temp_interval, headache_interval, age_interval):
        """
        Linguistic defuzzification: returns the output term (Low/Medium/High) most similar
        to the aggregated IT2 output set using Jaccard similarity.
        """
        self.temp_in.setInputMF(self._create_interval_input_mf("Temp", temp_interval, self.temp_in.getDomain()))
        self.headache_in.setInputMF(self._create_interval_input_mf("Head", headache_interval, self.headache_in.getDomain()))
        self.age_in.setInputMF(self._create_interval_input_mf("Age", age_interval, self.age_in.getDomain()))

        aggregated = None
        for rule in self.rulebase.getRules():
            firing = rule.getFStrength(self.rulebase.implicationMethod)
            if firing.getRight() <= 0.0:
                continue
            clipped = IntervalT2MF_Intersection(
                IntervalT2MF_Cylinder("FiringInterval", firing),
                rule.getConsequents()[0].getMembershipFunction()
            )
            aggregated = clipped if aggregated is None else IntervalT2MF_Union(aggregated, clipped)

        # If no rules fired, return "N/A" with zero similarities
        if aggregated is None:
            return "N/A", {"Low": 0.0, "Medium": 0.0, "High": 0.0}

        jaccard = JaccardSimilarity()
        n = self.urgency_out.getDiscretisationLevel()
        similarities = {
            label: jaccard.getSimilarity(aggregated, cons.getMembershipFunction(), n)
            for label, cons in self.con_urg.items()
        }

        best = max(similarities, key=similarities.get)
        return best, similarities