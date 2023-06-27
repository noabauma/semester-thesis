#!/usr/bin/env python3
import korali
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("model/")
from posteriorModel import *

# Defining Concurrent Jobs
cJobs = 1
solver = "Nested"
if (len(sys.argv) > 1): 
    cJobs = int(sys.argv[1])
    solver = str(sys.argv[2])

print("Number of workers: ", cJobs, " , Solver type: ", solver, flush=True)

path = "_result_" + str(cJobs).zfill(2) + "_"  + solver

# Make a boxplot
e = korali.Experiment()

e['Problem']['Type'] = 'Propagation'
e['Problem']['Execution Model'] = model

with open(path + '/latest') as f:
    d = json.load(f)

database = 'Sample Database' if solver == 'TMCMC' else 'Posterior Sample Database'

e['Variables'][0]['Name'] = "A_eps"
v = [p[0] for p in d['Results'][database]]
e['Variables'][0]['Precomputed Values'] = v

e['Variables'][1]['Name'] = "B"
v = [p[1] for p in d['Results'][database]]
e['Variables'][1]['Precomputed Values'] = v

e['Variables'][2]['Name'] = "SW_sigma"
v = [p[2] for p in d['Results'][database]]
e['Variables'][2]['Precomputed Values'] = v

e['Variables'][3]['Name'] = "rc"
v = [p[2] for p in d['Results'][database]]
e['Variables'][3]['Precomputed Values'] = v

e['Variables'][4]['Name'] = "sigma"
v = [p[3] for p in d['Results'][database]]
e['Variables'][4]['Precomputed Values'] = v

e['Solver']['Type'] = 'Executor'
e['Solver']['Executions Per Generation'] = 100

e['Console Output']['Verbosity'] = 'Minimal'
path += '_evalulation'
e['File Output']['Path'] = path
e['Store Sample Information'] = True

k = korali.Engine()
k.run(e)


#create boxplot
with open(path + '/latest') as f: d = json.load(f)

var ='Reference Evaluations'

N  = len(d['Samples'])
Ny = len(d['Samples'][0][var])

print("Number of samples = ", N)

y = np.empty((N,Ny), dtype=np.float64)
for k in range(N):
    y[k,:] = d['Samples'][k][var]

fig, axs = plt.subplots(1,4)

axs[0].boxplot(y[:,0])
axs[0].axhline(86.0, color='r', linestyle='--')
axs[0].title.set_text("WCA graphene [degree]")

axs[1].boxplot(y[:,1])
axs[1].axhline(106.925, color='r', linestyle='--')
axs[1].title.set_text("WCA CNT [degree]")

axs[2].boxplot(y[:,2])
axs[2].axhline(9000.0, color='r', linestyle='--')
axs[2].title.set_text("Friction coef. [kg/(m*s)]")

axs[3].boxplot(y[:,3])
axs[3].axhline(0.775, color='r', linestyle='--')
axs[3].title.set_text("Viscosity [mPa*s]")

plt.show()