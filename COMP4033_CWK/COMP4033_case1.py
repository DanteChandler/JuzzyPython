"""
COMP4033 Coursework - Case 1: All inputs are numeric

Summary:
- 3 Inputs: Temperature [0, 50], Headache [0, 10], Age [0, 130]
- 1 Output: Urgency [0, 100] (3 terms: Routine, Monitor, Emergency)
- MF Shapes: Trapezoidal at domain boundaries, Triangular for middle
- Rule Base: Exhaustive 64-rule grid (4 x 4 x 4)
- Operators: Minimum t-norm (AND), Maximum t-conorm (OR)
- Defuzzification: Centroid (Center of Gravity)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "JuzzyPython"))

import itertools # For generating all combinations of antecedent terms for the 64-rule base
from juzzyPython.generic.Tuple import Tuple
from juzzyPython.generic.Input import Input
from juzzyPython.generic.Output import Output
from juzzyPython.generic.Plot import Plot
from juzzyPython.type1.sets.T1MF_Triangular import T1MF_Triangular
from juzzyPython.type1.sets.T1MF_Trapezoidal import T1MF_Trapezoidal
from juzzyPython.type1.system.T1_Antecedent import T1_Antecedent
from juzzyPython.type1.system.T1_Consequent import T1_Consequent
from juzzyPython.type1.system.T1_Rule import T1_Rule
from juzzyPython.type1.system.T1_Rulebase import T1_Rulebase


class Case1FuzzyLogicSystem:
    def __init__(self):
        # Defining inputs and outputs with their ranges
        self.temp_in = Input("Temperature (°C)", Tuple(0.0, 50.0))
        self.headache_in = Input("Headache Severity", Tuple(0.0, 10.0))
        self.age_in = Input("Patient Age (Years)", Tuple(0.0, 130.0))
        self.urgency_out = Output("Urgency Score", Tuple(0.0, 100.0))
        self.urgency_out.setDiscretisationLevel(200)
        #Defining membership functions for each input and output variable
        
        # Temperature
        # Hypothermia (<35°C): NHS says anything below 35°C is a medical emergency
        # Normal (36.0–37.5°C): These are 'normal' body temperatures, with 36.8°C being the average
        # Moderate Fever (38.0–40.0°C): NHS says >38°C is a fever
        # Hyperpyrexia (>40.0°C): NHS says >40°C is a medical emergency
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
        self.age_adult = T1MF_Trapezoidal("Adult", [12.0, 18.0, 45.0, 50.0])
        self.age_older = T1MF_Trapezoidal("Older Adult (>50)", [45.0, 50.0, 130.0, 130.0])

        # Urgency (output)
        # Low: Home care or GP checkup, maybe even nothing
        # Medium: Monitor and follow up with GP or hospital
        # High: Immediate hospital referral or emergency services
        self.urg_low = T1MF_Trapezoidal("Low", [0.0, 0.0, 20.0, 40.0])
        self.urg_med = T1MF_Triangular("Medium", 25.0, 50.0, 75.0)
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

        # Building 64-rule base
        # Note: JuzzyPy's T1_Rulebase defaults to Minimum t-norm (0) and Maximum t-conorm (0) with Centroid defuzzification
        self.rulebase = T1_Rulebase()
        self._build_64_rules()

    def _determine_consequent(self, t_label, h_label, a_label):
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

    def _build_64_rules(self):
        """Generates all 4 x 4 x 4 = 64 combinations"""
        self.rules_log = [] # Logs each rule 
        rule_idx = 1 # Rule index for each log
        for t_label, h_label, a_label in itertools.product(
            self.ant_temp.keys(), self.ant_head.keys(), self.ant_age.keys() # All combinations of antecedent labels
        ):
            urg_key = self._determine_consequent(t_label, h_label, a_label) # Determine the consequent based on the rule logic
            # Creates the rule and add it to the rulebase
            rule = T1_Rule(
                [self.ant_temp[t_label], self.ant_head[h_label], self.ant_age[a_label]],
                consequent=self.con_urg[urg_key]
            )
            self.rulebase.addRule(rule) # Add the rule to the rulebase and log it
            # Logging the rule in a human-readable format for later display
            self.rules_log.append(
                f"R{rule_idx:02d}: IF Temp is {t_label:<14} AND Headache is {h_label:<13} "
                f"AND Age is {a_label:<17} THEN Urgency is {urg_key}"
            )
            rule_idx += 1

    def evaluate_patient(self, temp_val, headache_val, age_val, defuzz_type=1):
        """
        Sets numeric inputs and returns the defuzzified Urgency score
        temp_val: float - Temperature in °C
        headache_val: float - Headache severity on a scale of 0-10
        age_val: float - Patient age in years
        defuzz_type: int - 1 for Centroid defuzzification, 0 for Height defuzzification
        """
        self.temp_in.setInput(float(temp_val))
        self.headache_in.setInput(float(headache_val))
        self.age_in.setInput(float(age_val))

        # Evaluate using Centroid defuzzification (1 = Centroid, 0 = Height)
        output_dict = self.rulebase.evaluate(defuzz_type)
        return output_dict[self.urgency_out]

    def plot_membership_functions(self):
        """Plots the 4 sets of membership functions using JuzzyPy's Plot utility. Makes a graph"""
        plotter = Plot()

        variables = [
            ("Temperature Membership Functions", self.temp_in.getDomain(),
             [self.temp_hypo, self.temp_norm, self.temp_fever, self.temp_hyper]),
            ("Headache Severity Membership Functions", self.headache_in.getDomain(),
             [self.head_none, self.head_mild, self.head_mod, self.head_sev]),
            ("Patient Age Membership Functions", self.age_in.getDomain(),
             [self.age_u5, self.age_child, self.age_adult, self.age_older]),
            ("Urgency Output Membership Functions", self.urgency_out.getDomain(),
             [self.urg_low, self.urg_med, self.urg_high]),
        ]

        for title, domain, mf_list in variables:
            plotter.figure()
            for mf in mf_list:
                plotter.plotMF(
                    title,           # 1. Graph title
                    mf.getName(),    # 2. Legend label for this MF
                    mf,              # 3. The T1MF object itself
                    500,             # 4. Discretization steps
                    domain,          # 5. X-axis Tuple range
                    Tuple(0.0, 1.0), # 6. Y-axis Tuple range
                    False            # 7. addExtraEndPoints
                )
            plotter.legend()

        plotter.show()


# Execution section for demo and testing
if __name__ == "__main__":
    fls = Case1FuzzyLogicSystem()

    # Displaying the number of rules loaded and a header for the test cases
    print(f"COMP4033 CASE 1 (NUMERIC) MAMDANI FLS - {len(fls.rules_log)} RULES LOADED\n")

    # Example Test Cases 
    test_patients = [
        ("Patient 1 (Healthy Adult, Mild Headache)",       36.8, 2.0, 28.0),
        ("Patient 2 (Child 8yo, Moderate Fever & Pain)",   38.6, 5.0,  8.0),
        ("Patient 3 (Infant 2yo, High Fever & Distress)",  39.2, 6.0,  2.0),
        ("Patient 4 (Older Adult 68yo, Severe Headache)",  37.0, 8.5, 68.0),
        ("Patient 5 (Adult, Hyperpyrexia Crisis)",         41.2, 7.0, 35.0),
        ("Patient 6 (Elderly, Hypothermia Exposure)",      34.0, 1.0, 78.0),
    ]

    # Displaying the results of the test cases in a table
    print(f"{'Patient':<47} | {'Temp':<5} | {'Head':<5} | {'Age':<5} | {'Urgency (0-100)'}")
    print("-" * 85)
    for name, t, h, a in test_patients:
        score = fls.evaluate_patient(t, h, a)
        print(f"{name:<47} | {t:>5.1f} | {h:>5.1f} | {a:>5.1f} | {score:>10.2f}")
    print("=" * 85)

    # Prints full 64-rule table or launch plots
    ans = input("\nPrint all 64 rules to terminal? [y/n]: ").strip().lower()
    if ans == "y":
        for line in fls.rules_log:
            print(line)

    # Prints the Interval Type-2 Membership Function plots for all inputs and outputs
    plot_ans = input("Display Membership Function Plots? [y/n]: ").strip().lower()
    if plot_ans == "y":
        fls.plot_membership_functions()