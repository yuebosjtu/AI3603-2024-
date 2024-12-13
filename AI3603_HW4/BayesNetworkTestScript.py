# -*- coding:utf-8 -*-

from BayesianNetworks import *
import numpy as np
import pandas as pd

#############################
## Example Tests from Bishop `Pattern Recognition and Machine Learning` textbook on page 377
#############################
BatteryState = readFactorTable(['battery'], [0.9, 0.1], [[1, 0]])
FuelState = readFactorTable(['fuel'], [0.9, 0.1], [[1, 0]])
GaugeBF = readFactorTable(['gauge', 'battery', 'fuel'], [0.8, 0.2, 0.2, 0.1, 0.2, 0.8, 0.8, 0.9], [[1, 0], [1, 0], [1, 0]])

carNet = [BatteryState, FuelState, GaugeBF]  # carNet is a list of factors
## Notice that different order of operations give the same answer
## (rows/columns may be permuted)
joinFactors(joinFactors(BatteryState, FuelState), GaugeBF)
joinFactors(joinFactors(GaugeBF, FuelState), BatteryState)

marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'gauge')
joinFactors(marginalizeFactor(GaugeBF, 'gauge'), BatteryState)

joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState)
marginalizeFactor(joinFactors(joinFactors(GaugeBF, FuelState), BatteryState), 'battery')

marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'gauge')
marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'fuel')

evidenceUpdateNet(carNet, ['fuel', 'battery'], [1, 0])

# inference
print("inference starts")
print(inference(carNet, ['battery', 'fuel'], [], []))  ## chapter 8 equation (8.30)
print(inference(carNet, ['battery'], ['fuel'], [0]))  ## chapter 8 equation (8.31)
print(inference(carNet, ['battery'], ['gauge'], [0]))  ##chapter 8 equation  (8.32)
print(inference(carNet, [], ['gauge', 'battery'], [0, 0]))  ## chapter 8 equation (8.33)
print("inference ends")
###########################################################################
# RiskFactor Data Tests
###########################################################################
riskFactorNet = pd.read_csv('RiskFactorsData.csv')

# Create factors

income = readFactorTablefromData(riskFactorNet, ['income'])
smoke = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
exercise = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
long_sit = readFactorTablefromData(riskFactorNet, ['long_sit', 'income'])
stay_up = readFactorTablefromData(riskFactorNet, ['stay_up', 'income'])
bmi = readFactorTablefromData(riskFactorNet, ['bmi', 'income'])
diabetes = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi'])

## you need to create more factor tables

risk_net = [income, smoke, long_sit, stay_up, exercise, bmi, diabetes]
print("income dataframe is ")
print(income)
factors = riskFactorNet.columns

# example test p(diabetes|smoke=1,exercise=2,long_sit=1)

margVars = list(set(factors) - {'diabetes', 'smoke', 'exercise', 'long_sit'})
obsVars = ['smoke', 'exercise', 'long_sit']
obsVals = [1, 2, 1]

p = inference(risk_net, margVars, obsVars, obsVals)
print(p)


###########################################################################
# Please write your own test script
# HW4 test scripts start from here
###########################################################################

riskFactorNet = pd.read_csv('RiskFactorsData.csv')

income = readFactorTablefromData(riskFactorNet, ['income'])
smoke = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
exercise = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
long_sit = readFactorTablefromData(riskFactorNet, ['long_sit', 'income'])
stay_up = readFactorTablefromData(riskFactorNet, ['stay_up', 'income'])
bmi = readFactorTablefromData(riskFactorNet, ['bmi', 'exercise', 'long_sit', 'income'])
bp = readFactorTablefromData(riskFactorNet, ['bp', 'smoke', 'exercise', 'long_sit', 'stay_up', 'income'])
cholesterol = readFactorTablefromData(riskFactorNet, ['cholesterol', 'exercise', 'stay_up', 'smoke', 'income'])
stroke = readFactorTablefromData(riskFactorNet, ['stroke', 'bp', 'bmi', 'cholesterol'])
attack = readFactorTablefromData(riskFactorNet, ['attack', 'bp', 'bmi', 'cholesterol'])
angina = readFactorTablefromData(riskFactorNet, ['angina', 'bp', 'bmi', 'cholesterol'])
diabetes = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi'])

risk_net = [income, smoke, long_sit, stay_up, exercise, bmi, diabetes, cholesterol, stroke, attack, angina, bp]
factors = riskFactorNet.columns

## Question1
size = 1
for label in riskFactorNet.columns[1:]:
    size *= len(readFactorTablefromData(riskFactorNet, [label]))
print(size)

## Question2
health_outcomes = ['diabetes', 'stroke', 'attack', 'angina']
habits = ['smoke', 'exercise', 'long_sit', 'stay_up']
health_condition = ['bp', 'cholesterol', 'bmi']
bad_habits_var = [1, 2, 1, 1]
good_habits_var = [2, 1, 2, 2]
poor_health_var = [1, 1, 3]
good_health_var = [3, 2, 2]
hiddenVars_habits = set(factors) - {'stay_up', 'smoke', 'exercise', 'long_sit'}
hiddenVars_health = set(factors) - {'bp', 'cholesterol', 'bmi'}


# 2(a)
for outcome in health_outcomes:
    hidden_vars_set = hiddenVars_habits - {outcome}
    hidden_vars_lst = list(hidden_vars_set)

    bad_habits_prob = inference(risk_net, hidden_vars_lst, habits, bad_habits_var)
    print(f"Probability of {outcome} with bad habits:\n {bad_habits_prob}")

for outcome in health_outcomes:
    hidden_vars_set = hiddenVars_habits - {outcome}
    hidden_vars_lst = list(hidden_vars_set)

    good_habits_prob = inference(risk_net, hidden_vars_lst, habits, good_habits_var)
    print(f"Probability of {outcome} with good habits:\n {good_habits_prob}")

# 2(b)
for outcome in health_outcomes:
    hidden_vars_set = hiddenVars_health - {outcome}
    hidden_vars_lst = list(hidden_vars_set)

    poor_health_prob = inference(risk_net, hidden_vars_lst, health_condition, poor_health_var)
    print(f"Probability of {outcome} with bad health:\n {poor_health_prob}")

for outcome in health_outcomes:
    hidden_vars_set = hiddenVars_health - {outcome}
    hidden_vars_lst = list(hidden_vars_set)

    good_health_prob = inference(risk_net, hidden_vars_lst, health_condition, good_health_var)
    print(f"Probability of {outcome} with good health:\n {good_health_prob}")


## Question3
import matplotlib.pyplot as plt

outcomes = ["diabetes", "stroke", "attack", "angina"]
fig, ax = plt.subplots(2, 2, figsize=(14, 12))

for i, outcome in enumerate(outcomes):
    probs = []
    for j in range(1, 9):
        hidden = [x for x in factors if x != "income" and x != outcome]
        p = inference(risk_net, hidden, ["income"], [j])
        print(p)
        tmp = p[p[outcome] == 1]["probs"].values[0]
        probs.append(tmp)
    
    # Select the appropriate subplot
    row = i // 2
    col = i % 2
    ax[row, col].plot(range(1, 9), probs, "ro-")
    ax[row, col].set_title(f"probabilities of {outcome}")
    ax[row, col].set_xlabel("income levels")
    ax[row, col].set_ylabel("probs")

plt.tight_layout()
plt.show()

## Question4
income_2 = readFactorTablefromData(riskFactorNet, ["income"])
exercise_2 = readFactorTablefromData(riskFactorNet, ["exercise", "income"])
long_sit_2 = readFactorTablefromData(riskFactorNet, ["long_sit", "income"])
stay_up_2 = readFactorTablefromData(riskFactorNet, ["stay_up", "income"])
smoke_2 = readFactorTablefromData(riskFactorNet, ["smoke", "income"])
bmi_2 = readFactorTablefromData(riskFactorNet, ["bmi", "exercise", "income", "long_sit"])
bp_2 = readFactorTablefromData(riskFactorNet, ["bp", "exercise", "long_sit", "income", "stay_up", "smoke"])
cholesterol_2 = readFactorTablefromData(riskFactorNet, ["cholesterol", "exercise", "stay_up", "income", "smoke"])
diabetes_2 = readFactorTablefromData(riskFactorNet, ["diabetes", "bmi", "smoke", "exercise"])
stroke_2 = readFactorTablefromData(riskFactorNet, ["stroke", "bmi", "bp", "cholesterol", "smoke", "exercise"])
attack_2 = readFactorTablefromData(riskFactorNet, ["attack", "bmi", "bp", "cholesterol", "smoke", "exercise"])
angina_2 = readFactorTablefromData(riskFactorNet, ["angina", "bmi", "bp", "cholesterol", "smoke", "exercise"])

risk_net_2 = [income_2, exercise_2, long_sit_2, stay_up_2, smoke_2, bmi_2, bp_2, cholesterol_2, diabetes_2, stroke_2, attack_2, angina_2]

health_outcomes = ['diabetes', 'stroke', 'attack', 'angina']
habits = ['smoke', 'exercise', 'long_sit', 'stay_up']
health_condition = ['bp', 'cholesterol', 'bmi']
bad_habits_var = [1, 2, 1, 1]
good_habits_var = [2, 1, 2, 2]
poor_health_var = [1, 1, 3]
good_health_var = [3, 2, 2]
hiddenVars_habits = set(factors) - {'stay_up', 'smoke', 'exercise', 'long_sit'}
hiddenVars_health = set(factors) - {'bp', 'cholesterol', 'bmi'}


# 4(a)
for outcome in health_outcomes:
    hidden_vars_set = hiddenVars_habits - {outcome}
    hidden_vars_lst = list(hidden_vars_set)

    bad_habits_prob = inference(risk_net_2, hidden_vars_lst, habits, bad_habits_var)
    print(f"Probability of {outcome} with bad habits:\n {bad_habits_prob}")

for outcome in health_outcomes:
    hidden_vars_set = hiddenVars_habits - {outcome}
    hidden_vars_lst = list(hidden_vars_set)

    good_habits_prob = inference(risk_net_2, hidden_vars_lst, habits, good_habits_var)
    print(f"Probability of {outcome} with good habits:\n {good_habits_prob}")

# 4(b)
for outcome in health_outcomes:
    hidden_vars_set = hiddenVars_health - {outcome}
    hidden_vars_lst = list(hidden_vars_set)

    poor_health_prob = inference(risk_net_2, hidden_vars_lst, health_condition, poor_health_var)
    print(f"Probability of {outcome} with bad health:\n {poor_health_prob}")

for outcome in health_outcomes:
    hidden_vars_set = hiddenVars_health - {outcome}
    hidden_vars_lst = list(hidden_vars_set)

    good_health_prob = inference(risk_net_2, hidden_vars_lst, health_condition, good_health_var)
    print(f"Probability of {outcome} with good health:\n {good_health_prob}")

## Question5
income_3 = readFactorTablefromData(riskFactorNet, ["income"])
exercise_3 = readFactorTablefromData(riskFactorNet, ["exercise", "income"])
long_sit_3 = readFactorTablefromData(riskFactorNet, ["long_sit", "income"])
stay_up_3 = readFactorTablefromData(riskFactorNet, ["stay_up", "income"])
smoke_3 = readFactorTablefromData(riskFactorNet, ["smoke", "income"])
bmi_3 = readFactorTablefromData(riskFactorNet, ["bmi", "exercise", "income", "long_sit"])
bp_3 = readFactorTablefromData(riskFactorNet, ["bp", "exercise", "long_sit", "income", "stay_up", "smoke"])
cholesterol_3 = readFactorTablefromData(riskFactorNet, ["cholesterol", "exercise", "stay_up", "income", "smoke"])
diabetes_3 = readFactorTablefromData(riskFactorNet, ["diabetes", "bmi", "smoke", "exercise"])
stroke_3 = readFactorTablefromData(riskFactorNet, ["stroke", "bmi", "bp", "cholesterol", "smoke", "exercise", "diabetes"])
attack_3 = readFactorTablefromData(riskFactorNet, ["attack", "bmi", "bp", "cholesterol", "smoke", "exercise"])
angina_3 = readFactorTablefromData(riskFactorNet, ["angina", "bmi", "bp", "cholesterol", "smoke", "exercise"])

# Create the new Bayesian Network
risk_net_3 = [income_3, exercise_3, long_sit_3, stay_up_3, smoke_3, bmi_3, bp_3, cholesterol_3, diabetes_3, stroke_3, attack_3, angina_3]

query = "stroke"
hidden = [x for x in factors if x != query and x != "diabetes"]
p_2 = inference(risk_net_2, hidden, ["diabetes"], [1])
p_3 = inference(risk_net_3, hidden, ["diabetes"], [1])
res_2 = p_2[p_2[query] == 1]["probs"].values[0]
res_3 = p_3[p_3[query] == 1]["probs"].values[0]
print(f"P(stroke = 1|diabetes = 1) in Network 2 is {res_2}")
print(f"P(stroke = 1|diabetes = 1) in Network 3 is {res_3}")

query = "stroke"
hidden = [x for x in factors if x != query and x != "diabetes"]
p_2 = inference(risk_net_2, hidden, ["diabetes"], [3])
p_3 = inference(risk_net_3, hidden, ["diabetes"], [3])
res_2 = p_2[p_2[query] == 1]["probs"].values[0]
res_3 = p_3[p_3[query] == 1]["probs"].values[0]
print(f"P(stroke = 1|diabetes = 3) in Network 2 is {res_2}")
print(f"P(stroke = 1|diabetes = 3) in Network 3 is {res_3}")