"""
COMP4033 Coursework - Case 2: All inputs are interval-valued

Summary:
- System Type: Interval Type-2 Mamdani Fuzzy Logic System with Non-Singleton Interval Inputs
- 3 Inputs: Temperature [0, 50], Headache [0, 10], Age [0, 130]
- 1 Output: Urgency [0, 100] (3 terms: Low, Medium, High)
- MF Shapes: Interval Type-2 Trapezoidal at domain boundaries/plateaus, IT2 Triangular for middle (with UMF and LMF)
- Rule Base: 21 rules
- Operators: Minimum t-norm (AND), Maximum t-conorm (OR)
- Type-Reduction & Defuzzification: Center-of-Sets (COS) / Karnik-Mendel Type-Reduction -> Centroid
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "JuzzyPython"))

import itertools 
from juzzyPython.generic.Tuple import Tuple
from juzzyPython.generic.Input import Input
from juzzyPython.generic.Output import Output
from juzzyPython.generic.Plot import Plot
from juzzyPython.type1.sets.T1MF_Triangular import T1MF_Triangular
from juzzyPython.type1.sets.T1MF_Trapezoidal import T1MF_Trapezoidal
from juzzyPython.intervalType2.sets.IntervalT2MF_Triangular import IntervalT2MF_Triangular as IT2MF_Triangular
from juzzyPython.intervalType2.sets.IntervalT2MF_Trapezoidal import IntervalT2MF_Trapezoidal as IT2MF_Trapezoidal
from juzzyPython.intervalType2.system.IT2_Antecedent import IT2_Antecedent
from juzzyPython.intervalType2.system.IT2_Consequent import IT2_Consequent
from juzzyPython.intervalType2.system.IT2_Rule import IT2_Rule
from juzzyPython.intervalType2.system.IT2_Rulebase import IT2_Rulebase


class Case2FuzzyLogicSystem:

    RULES = [
            #A severe symptom such as Hypothermia, Hyperpyrexia or Severe Pain automatically makes the consequent high urgency
            ("Hypothermia", None, None, "High"),
            ("Hyperpyrexia", None, None, "High"),
            (None, "Severe Pain", None, "High"),


            ("Normal", "No Pain", None, "Low"),
            ("Normal", "Mild Pain", None, "Low"),
            #One moderate symptom makes the consequent low for adults but medium for Under 5s, children and Older Adults
            ("Moderate Fever", "No Pain","Under 5", "Medium"),
            ("Moderate Fever", "No Pain","Child (5-12)", "Medium"),
            ("Moderate Fever", "No Pain","Adult", "Low"),
            ("Moderate Fever", "No Pain","Older Adult (>50)", "Medium"),

            ("Moderate Fever", "Mild Pain","Under 5", "Medium"),
            ("Moderate Fever", "Mild Pain","Child (5-12)", "Medium"),
            ("Moderate Fever", "Mild Pain","Adult", "Low"),
            ("Moderate Fever", "Mild Pain","Older Adult (>50)", "Medium"),

            ("Normal", "Moderate Pain", "Under 5" , "Medium"),
            ("Normal", "Moderate Pain", "Child (5-12)" , "Medium"),
            ("Normal", "Moderate Pain", "Adult" , "Low"),
            ("Normal", "Moderate Pain", "Older Adult (>50)" , "Medium"),

            #Two moderate symptoms makes the consequent High for under 5s, medium for children, medium for adults and high for older Adults
            ("Moderate Fever", "Moderate Pain", "Under 5", "High"),
            ("Moderate Fever", "Moderate Pain", "Child (5-12)", "Medium"),
            ("Moderate Fever", "Moderate Pain", "Adult" , "Medium"),
            ("Moderate Fever", "Moderate Pain", "Older Adult (>50)" , "High"),
        ]
  


    def __init__(self):
        # Defining inputs and outputs with their ranges 
        self.temp_in = Input("Temperature (°C)", Tuple(0, 50))
        self.headache_in = Input("Headache Severity", Tuple(0, 10))
        self.age_in = Input("Patient Age (Years)", Tuple(0, 130))
        self.urgency_out = Output("Urgency Score", Tuple(0.0, 100.0))
        self.urgency_out.setDiscretisationLevel(200)

        # Defining Interval Type-2 membership functions (requires an Upper MF and a Lower MF to form the FOU)
        
        # Temperature
        # Hypothermia (<35°C): UMF extends to 36.2°C for cautious boundary; LMF is stricter up to 34.8°C
        self.temp_hypo_umf = T1MF_Trapezoidal("Hypothermia_UMF", [0.0, 0.0, 35.0, 36.2])
        self.temp_hypo_lmf = T1MF_Trapezoidal("Hypothermia_LMF", [0.0, 0.0, 34.5, 35.5])
        self.temp_hypo = IT2MF_Trapezoidal("Hypothermia", self.temp_hypo_umf, self.temp_hypo_lmf)

        # Normal (36.0–37.5°C): Centered at 36.8°C
        self.temp_norm_umf = T1MF_Triangular("Normal_UMF", 35.0, 36.8, 38.2)
        self.temp_norm_lmf = T1MF_Triangular("Normal_LMF", 35.5, 36.8, 37.7)
        self.temp_norm = IT2MF_Triangular("Normal", self.temp_norm_umf, self.temp_norm_lmf)

        # Moderate Fever (38.0–40.0°C): Centered at 38.8°C
        self.temp_fever_umf = T1MF_Triangular("Moderate Fever_UMF", 37.3, 38.8, 40.2)
        self.temp_fever_lmf = T1MF_Triangular("Moderate Fever_LMF", 37.8, 38.8, 39.7)
        self.temp_fever = IT2MF_Triangular("Moderate Fever", self.temp_fever_umf, self.temp_fever_lmf)

        # Hyperpyrexia (>40.0°C): UMF starts ramping at 38.8°C; LMF ramps from 39.3°C to 40.3°C
        self.temp_hyper_umf = T1MF_Trapezoidal("Hyperpyrexia_UMF", [38.8, 40.0, 50.0, 50.0])
        self.temp_hyper_lmf = T1MF_Trapezoidal("Hyperpyrexia_LMF", [39.3, 40.3, 50.0, 50.0])
        self.temp_hyper = IT2MF_Trapezoidal("Hyperpyrexia", self.temp_hyper_umf, self.temp_hyper_lmf)

        # Headache [0 - 10]
        # Based on NHS Headache Severity Scale (0-10)
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

        # Age [0 - 130 Years]
        # Under 5 (<5), Child (5-12), Adult (13-50), Older Adult (>50, with 65+ high risk)
        self.age_u5_umf = T1MF_Trapezoidal("Under 5_UMF", [0.0, 0.0, 5.0, 6.5])
        self.age_u5_lmf = T1MF_Trapezoidal("Under 5_LMF", [0.0, 0.0, 4.5, 5.5])
        self.age_u5 = IT2MF_Trapezoidal("Under 5", self.age_u5_umf, self.age_u5_lmf)

        self.age_child_umf = T1MF_Triangular("Child (5-12)_UMF", 4.5, 8.5, 13.5)
        self.age_child_lmf = T1MF_Triangular("Child (5-12)_LMF", 5.5, 8.5, 12.5)
        self.age_child = IT2MF_Triangular("Child (5-12)", self.age_child_umf, self.age_child_lmf)

        self.age_adult_umf = T1MF_Trapezoidal("Adult_UMF", [11.5, 17.5, 46.0, 52.0])
        self.age_adult_lmf = T1MF_Trapezoidal("Adult_LMF", [12.5, 18.5, 44.0, 48.0])
        self.age_adult = IT2MF_Trapezoidal("Adult", self.age_adult_umf, self.age_adult_lmf)

        self.age_older_umf = T1MF_Trapezoidal("Older Adult (>50)_UMF", [43.0, 49.0, 130.0, 130.0])
        self.age_older_lmf = T1MF_Trapezoidal("Older Adult (>50)_LMF", [47.0, 52.0, 130.0, 130.0])
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

        # Building 21 rule base
        self.rulebase = IT2_Rulebase()
        self._build_rules()

    def _build_rules(self):
        self.rules_log = [] 
        for i, (t,h,a, urg) in enumerate (self.RULES, start =1):
            antecedents= []
            parts = []
            if t is not None:
                antecedents.append(self.ant_temp[t]); parts.append(f"Temp is {t}")
            if h is not None:
                antecedents.append(self.ant_head[h]); parts.append(f"Headache is {h}")
            if a is not None:
                antecedents.append(self.ant_age[a]); parts.append(f"Age is {a}")
            rule = IT2_Rule(antecedents, consequent = self.con_urg[urg])
            self.rulebase.addRule(rule)
            self.rules_log.append(f"R{i:02d}: IF " + " AND ".join(parts) + f" THEN Urgency is {urg}")



    def _create_interval_input_mf(self, name, interval_tuple, domain_tuple):
        """
        Converts an input interval [low, high] into a Non-Singleton Triangular Input MF
        spanning [low, mid, high] for JuzzyPy's native non-singleton IT2 fuzzification.
        """
        low, high = float(interval_tuple[0]), float(interval_tuple[1])
        d_min, d_max = float(domain_tuple.getLeft()), float(domain_tuple.getRight())
        
        # Makes sure the interval is within the domain bounds and calculates the midpoint
        a = max(d_min, low)
        c = min(d_max, high)
        b = (a + c) / 2.0

        return T1MF_Triangular(f"{name}_IntervalInput", a, b, c)

    def evaluate_patient_interval(self, temp_interval, headache_interval, age_interval, type_reduction=0):
        """
        Evaluates a patient whose Temperature, Headache, and Age are provided as intervals
        
        How Interval Inputs are Processed:
        1. Each input interval is converted into a Non-Singleton Triangular Input MF spanning [low, mid, high]
        2. The rulebase is evaluated using JuzzyPy's IT2 evaluation
        3. The defuzzified Urgency score and the Tuple are returned
        4. The Tuple represents the range of possible Urgency scores given the input intervals, capturing the uncertainty in the inputs
        5. The type_reduction parameter allows selection of the type-reduction method
        """
        # Setting the input MFs for the rulebase evaluation
        self.temp_in.setInputMF(self._create_interval_input_mf("Temp", temp_interval, self.temp_in.getDomain()))
        self.headache_in.setInputMF(self._create_interval_input_mf("Head", headache_interval, self.headache_in.getDomain()))
        self.age_in.setInputMF(self._create_interval_input_mf("Age", age_interval, self.age_in.getDomain()))

        # JuzzyPy IT2 evaluation: returns both the defuzzified score and the [y_left, y_right] Tuple
        centroid_dict = self.rulebase.evaluateGetCentroid(type_reduction)
        tr_list = centroid_dict[self.urgency_out] # Format in JuzzyPy: [Tuple(y_left, y_right), defuzzified_float]

        tr_tuple = tr_list[0]
        defuzz_score = (tr_tuple.getLeft() + tr_tuple.getRight())/ 2.0
        return defuzz_score, (float(tr_tuple.getLeft()), float(tr_tuple.getRight()))

    def plot_membership_functions(self):
        """Plots the 4 sets of Interval Type-2 membership functions (Upper & Lower MFs) using JuzzyPy's Plot utility."""
        plotter = Plot()

        variables = [
            ("IT2 Temperature Membership Functions (UMF & LMF)", self.temp_in.getDomain(),
             [self.temp_hypo, self.temp_norm, self.temp_fever, self.temp_hyper]),
            ("IT2 Headache Severity Membership Functions (UMF & LMF)", self.headache_in.getDomain(),
             [self.head_none, self.head_mild, self.head_mod, self.head_sev]),
            ("IT2 Patient Age Membership Functions (UMF & LMF)", self.age_in.getDomain(),
             [self.age_u5, self.age_child, self.age_adult, self.age_older]),
            ("IT2 Urgency Output Membership Functions (UMF & LMF)", self.urgency_out.getDomain(),
             [self.urg_low, self.urg_med, self.urg_high]),
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



# Execution section for demo and testing
if __name__ == "__main__":
    fls = Case2FuzzyLogicSystem()

    # Displaying the number of rules loaded and a header for the test cases
    print(f"COMP4033 CASE 2 (INTERVAL / IT2) MAMDANI FLS - {len(fls.rules_log)} RULES LOADED\n")

    # Example Interval Test Cases
    # Note: Each interval is centered on the exact same midpoint as the Case 1 test patients
    test_patients_intervals = [
        ("Patient 1 (Healthy Adult, Mild Headache)",      (36.4, 37.2), (1.0, 3.0), (26.0, 30.0)), 
        ("Patient 2 (Child ~8yo, Mod Fever & Pain)",      (38.1, 39.1), (4.0, 6.0), ( 7.0,  9.0)), 
        ("Patient 3 (Infant ~2yo, High Fever & Distress)",(38.7, 39.7), (5.0, 7.0), ( 1.5,  2.5)), 
        ("Patient 4 (Older Adult ~68yo, Severe Headache)",(36.5, 37.5), (7.5, 9.5), (65.0, 71.0)), 
        ("Patient 5 (Adult, Hyperpyrexia Crisis)",        (40.7, 41.7), (6.0, 8.0), (32.0, 38.0)), 
        ("Patient 6 (Elderly, Hypothermia Exposure)",     (33.5, 34.5), (0.5, 1.5), (75.0, 81.0)),
    ]

    # Displaying the results of the interval test cases in a table
    print(f"{'Patient':<46} | {'Temp [C]':<12} | {'Head [0-10]':<11} | {'Age [Yrs]':<11} | {'Urgency'} | {'Urgency Range'}")
    print("-" * 118)
    for name, t_int, h_int, a_int in test_patients_intervals:
        score, (u_min, u_max) = fls.evaluate_patient_interval(t_int, h_int, a_int)
        t_str = f"[{t_int[0]:.1f},{t_int[1]:.1f}]"
        h_str = f"[{h_int[0]:.1f},{h_int[1]:.1f}]"
        a_str = f"[{a_int[0]:.1f},{a_int[1]:.1f}]"
        u_range_str = f"[{u_min:>5.2f}, {u_max:>5.2f}]"
        print(f"{name:<46} | {t_str:<12} | {h_str:<11} | {a_str:<11} | {score:>7.2f} | {u_range_str}")
    print("=" * 118)

    # Prints full 21-rule table or launch plots
    ans = input(f"\nPrint all {len(fls.rules_log)} rules to terminal? [y/n]: ").strip().lower()
    if ans == "y":
        for line in fls.rules_log:
            print(line)

    # Prints the Interval Type-2 Membership Function plots for all inputs and outputs
    plot_ans = input("Display IT2 Membership Function Plots? [y/n]: ").strip().lower()
    if plot_ans == "y":
        fls.plot_membership_functions()