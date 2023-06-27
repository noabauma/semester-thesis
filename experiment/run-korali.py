#!/usr/bin/env python3
import korali
import sys

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
e["Problem"]["Computational Model"] = model

e["Distributions"][0]["Name"] = "Uniform 0"             #A_eps = A*epsilon
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] =  10.0 * 0.01         #6.947695457055374e-21 = 4184.0/6.0221476e23 [J*mol/kcal]
e["Distributions"][0]["Maximum"] = 100.0 * 0.20 

e["Distributions"][1]["Name"] = "Uniform 1"             #B
e["Distributions"][1]["Type"] = "Univariate/Uniform"
e["Distributions"][1]["Minimum"] = 0.0
e["Distributions"][1]["Maximum"] = 1.0

e["Distributions"][2]["Name"] = "Uniform 2"             #sigma
e["Distributions"][2]["Type"] = "Univariate/Uniform"
e["Distributions"][2]["Minimum"] = 2.0
e["Distributions"][2]["Maximum"] = 5.0

e["Distributions"][3]["Name"] = "Uniform 3"             #rc
e["Distributions"][3]["Type"] = "Univariate/Uniform"
e["Distributions"][3]["Minimum"] = 2.5
e["Distributions"][3]["Maximum"] = 12.0

e["Distributions"][4]["Name"] = "Uniform 4"
e["Distributions"][4]["Type"] = "Univariate/Uniform"
e["Distributions"][4]["Minimum"] = 0.0
e["Distributions"][4]["Maximum"] = 5.0

# Configuring the problem's variables and their prior distributions
e["Variables"][0]["Name"] = "A_eps"                     #[J]
e["Variables"][0]["Prior Distribution"] = "Uniform 0"

e["Variables"][1]["Name"] = "B"                         #[1]
e["Variables"][1]["Prior Distribution"] = "Uniform 1"

e["Variables"][2]["Name"] = "SW_sigma"                  #[angstrom]
e["Variables"][2]["Prior Distribution"] = "Uniform 2"

e["Variables"][3]["Name"] = "rc"                        #[angstrom]
e["Variables"][3]["Prior Distribution"] = "Uniform 3"

e["Variables"][4]["Name"] = "[Sigma]"
e["Variables"][4]["Prior Distribution"] = "Uniform 4"

# Configuring Solver parameters
if solver == 'TMCMC':
    e["Solver"]["Type"] = "Sampler/TMCMC"
    e["Solver"]["Population Size"] = 10000
    e["Solver"]["Termination Criteria"]["Max Generations"] = 32
    e["Solver"]["Termination Criteria"]["Target Annealing Exponent"] = 2.0
else:
    e['Solver']['Type'] = 'Sampler/Nested'
    e['Solver']['Batch Size'] = cJobs
    e['Solver']['Number Live Points'] = 5000
    e['Solver']['Resampling Method'] = 'Ellipse'
    e["Solver"]["Termination Criteria"]["Max Generations"] = 5000
    #e['Solver']["Termination Criteria"]['Min Log Evidence Delta'] = 1.0
    e['File Output']['Frequency'] = 1000
    e['Console Output']['Frequency'] = 1000

# General Settings
path = "_result_" + str(cJobs).zfill(2) + "_"  + solver
e["File Output"]["Path"] = path
e["Console Output"]["Verbosity"] = "Detailed"

# Selecting external conduit
k["Conduit"]["Type"] = "Concurrent"
k["Conduit"]["Concurrent Jobs"] = cJobs

k.run(e)
