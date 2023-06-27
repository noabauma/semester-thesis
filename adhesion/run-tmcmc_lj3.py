#!/usr/bin/env python3
import korali
import sys
import json

sys.path.append("model/")
from posteriorModel import *

k = korali.Engine()
e = korali.Experiment()

# Defining Concurrent Jobs
cJobs = 1
solver = "Nested"
if (len(sys.argv) > 1): 
    cJobs = int(sys.argv[1])
    solver = str(sys.argv[2])

print("Number of workers: ", cJobs, " , Solver type: ", solver, flush=True)


# Setting up the reference likelihood for the Bayesian Problem
e["Problem"]["Type"] = "Bayesian/Reference"
e["Problem"]["Likelihood Model"] = "Normal"
e["Problem"]["Reference Data"] = getReferenceData()
e["Problem"]["Computational Model"] = model_lj3

e["Distributions"][0]["Name"] = "Uniform 0"             #eps
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = 0.01                  #6.947695457055374e-21 = 4184.0/6.02214076e23 [J*mol/kcal]
e["Distributions"][0]["Maximum"] = 0.15 

e["Distributions"][1]["Name"] = "Uniform 1"             #rc
e["Distributions"][1]["Type"] = "Univariate/Uniform"
e["Distributions"][1]["Minimum"] = 5.0
e["Distributions"][1]["Maximum"] = 15.0

e["Distributions"][2]["Name"] = "Uniform 2"             #[Sigma]
e["Distributions"][2]["Type"] = "Univariate/Uniform"
e["Distributions"][2]["Minimum"] = 0.0
e["Distributions"][2]["Maximum"] = 5.0

# Configuring the problem's variables and their prior distributions
e["Variables"][0]["Name"] = "eps"                           #[J]
e["Variables"][0]["Prior Distribution"] = "Uniform 0"

e["Variables"][1]["Name"] = "rc"                      #[angstrom]
e["Variables"][1]["Prior Distribution"] = "Uniform 1"

e["Variables"][2]["Name"] = "[Sigma]"                      
e["Variables"][2]["Prior Distribution"] = "Uniform 2"


# Configuring Solver parameters
if solver == 'CMAES':
    for i in range(5):
        L = e['Distributions'][i]['Maximum'] - e['Distributions'][i]['Minimum']
        e['Variables'][i]['Initial Value'] =  e['Distributions'][i]['Minimum'] + 0.5*L
        e['Variables'][i]['Initial Standard Deviation'] = 0.5*L

    e['Solver']['Type'] = 'Optimizer/CMAES'
    e['Solver']['Population Size'] = 1000
    e['Solver']['Termination Criteria']['Min Value Difference Threshold'] = 1e-32
    e['Solver']['Termination Criteria']['Max Generations'] = 2000
    e['Console Output']['Frequency'] = 100
elif solver == 'TMCMC':
    e["Solver"]["Type"] = "Sampler/TMCMC"
    e["Solver"]["Population Size"] = 10000
    e["Solver"]["Termination Criteria"]["Max Generations"] = 32
    e["Solver"]["Termination Criteria"]["Target Annealing Exponent"] = 2.0
else:
    e['Solver']['Type'] = 'Sampler/Nested'
    e['Solver']['Batch Size'] = cJobs
    e['Solver']['Number Live Points'] = 2000
    e['Solver']['Resampling Method'] = 'Ellipse'
    e["Solver"]["Termination Criteria"]["Max Generations"] = 16000
    #e['Solver']["Termination Criteria"]['Min Log Evidence Delta'] = 1.0
    e['File Output']['Frequency'] = 1000
    e['Console Output']['Frequency'] = 1000

# General Settings
path = "_result_lj3_" + str(cJobs).zfill(2) + "_"  + solver
e["File Output"]["Path"] = path
e["Console Output"]["Verbosity"] = "Detailed"

# Selecting external conduit
k["Conduit"]["Type"] = "Concurrent"
k["Conduit"]["Concurrent Jobs"] = cJobs

k.run(e)


# Make a boxplot
e = korali.Experiment()

e['Problem']['Type'] = 'Propagation'
e['Problem']['Execution Model'] = model_lj3

with open(path + '/latest') as f:
    d = json.load(f)

database = 'Sample Database' if solver == 'TMCMC' else 'Posterior Sample Database'

e['Variables'][0]['Name'] = "eps"
v = [p[0] for p in d['Results'][database]]
e['Variables'][0]['Precomputed Values'] = v

e['Variables'][1]['Name'] = "LJ_sigma"
v = [p[1] for p in d['Results'][database]]
e['Variables'][1]['Precomputed Values'] = v

e['Variables'][2]['Name'] = "sigma"
v = [p[2] for p in d['Results'][database]]
e['Variables'][2]['Precomputed Values'] = v

e['Solver']['Type'] = 'Executor'
e['Solver']['Executions Per Generation'] = 100

e['Console Output']['Verbosity'] = 'Minimal'
path += '_evalulation'
e['File Output']['Path'] = path
e['Store Sample Information'] = True

k = korali.Engine()
k.run(e)

# Uncomment the next two lines to get a boxplot of the evaluations
from plots import *
box_plot( path + '/latest')
