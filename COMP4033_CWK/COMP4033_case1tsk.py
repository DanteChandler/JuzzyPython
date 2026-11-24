"""
COMP4033 Coursework - Case 1: All inputs are numeric

Summary:


"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "JuzzyPython"))

from juzzyPython.generic.Tuple import Tuple
from juzzyPython.generic.Input import Input
from juzzyPython.generic.Plot import Plot
from juzzyPython.type1.sets.T1MF_Triangular import T1MF_Triangular
from juzzyPython.type1.sets.T1MF_Trapezoidal import T1MF_Trapezoidal
from juzzyPython.type1.system.T1_Antecedent import T1_Antecedent



class Case1TSKFuzzyLogicSystem:
  
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

    TSK_Constants = {"Low": 10.0, "Medium": 50.0, "High": 90.0}
    
    
    def __init__(self):



        # Defining inputs and outputs with their ranges
        self.temp_in = Input("Temperature (°C)", Tuple(0.0, 50.0))
        self.headache_in = Input("Headache Severity", Tuple(0.0, 10.0))
        self.age_in = Input("Patient Age (Years)", Tuple(0.0, 130.0))
       

        #Defining inputs
        
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
        self.rules_log = [] 
        for i, (t,h,a, urg) in enumerate (self.RULES, start =1):
            parts = []
            if t is not None:
                parts.append(f"Temp is {t}")
            if h is not None:
                parts.append(f"Headache is {h}")
            if a is not None:
               parts.append(f"Age is {a}")
            self.rules_log.append(f"R{i:02d}: IF " +  " AND ". join(parts) + f" THEN Urgency = {self.TSK_Constants[urg]:.0f} ({urg})")



    def evaluate_patient(self, temp_val, headache_val, age_val):
        """
        Zeroth order TSK output:
        """
        self.temp_in.setInput(float(temp_val))
        self.headache_in.setInput(float(headache_val))
        self.age_in.setInput(float(age_val))

        top = 0.0
        bottom = 0.0
        for(t,h,a,urg) in self.RULES:
            memberships = []
            if t is not None:
                memberships.append(self.ant_temp[t].getFS())
            if h is not None:
                memberships.append(self.ant_head[h].getFS())
            if a is not None:
                memberships.append(self.ant_age[a].getFS())
            firing = min(memberships)
            top = top + firing * self.TSK_Constants[urg]
            bottom = bottom + firing
        if bottom == 0:
            return None
        return top/ bottom


    def plot_membership_functions(self):
        """Plots the input membership functions"""
        plotter = Plot()

        variables = [
            ("Temperature Membership Functions", self.temp_in.getDomain(),
             [self.temp_hypo, self.temp_norm, self.temp_fever, self.temp_hyper]),
            ("Headache Severity Membership Functions", self.headache_in.getDomain(),
             [self.head_none, self.head_mild, self.head_mod, self.head_sev]),
            ("Patient Age Membership Functions", self.age_in.getDomain(),
             [self.age_u5, self.age_child, self.age_adult, self.age_older]),
           
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
    fls = Case1TSKFuzzyLogicSystem()

    # Displaying the number of rules loaded and a header for the test cases
    print(f"COMP4033 CASE 1 (NUMERIC) ZEROTH-ORDER TSK - {len(fls.rules_log)} RULES LOADED\n")

    # Example Test Cases 
    test_patients = [
        ("Patient 1 (Healthy Adult, Mild Headache)",       36.8, 2.0, 28.0),
        ("Patient 2 (Child 8yo, Moderate Fever & Pain)",   38.6, 5.0,  8.0),
        ("Patient 3 (Infant 2yo, High Fever & Distress)",  39.2, 6.0,  2.0),
        ("Patient 4 (Older Adult 68yo, Severe Headache)",  37.0, 8.5, 68.0),
        ("Patient 5 (Adult, Hyperpyrexia Crisis)",         41.2, 7.0, 35.0),
        ("Patient 6 (Elderly, Hypothermia Exposure)",      34.0, 1.0, 78.0),
        ("Patient 7 (Adult, Borderline Temp, Mod Pain)",   37.8, 5.0, 30.0),
    ]

    # Displaying the results of the test cases in a table
    print(f"{'Patient':<47} | {'Temp':<5} | {'Head':<5} | {'Age':<5} | {'Urgency (0-100)'}")
    print("-" * 85)
    for name, t, h, a in test_patients:
        score = fls.evaluate_patient(t, h, a)
        print(f"{name:<47} | {t:>5.1f} | {h:>5.1f} | {a:>5.1f} | {score:>10.2f}")
    print("=" * 85)

    # Prints full rule table 
    ans = input(f"\nPrint all {len(fls.rules_log)} rules to terminal? [y/n]: ").strip().lower()
    if ans == "y":
        for line in fls.rules_log:
            print(line)

    # Prints the Interval Type-1 Membership Function plots for all inputs and outputs
    plot_ans = input("Display Membership Function Plots? [y/n]: ").strip().lower()
    if plot_ans == "y":
        fls.plot_membership_functions()