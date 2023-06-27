#! /usr/bin/env python3
import json
import numpy as np
import sys
from sklearn import preprocessing

#this script decides which parameters are the best

if __name__ == "__main__":
    #goal parameters
    goal = np.array([260.0, 0.0]).reshape((1, -1))

    if len(sys.argv) < 2:
        print("Give me the path you want", flush=True)
        exit(1)

    file = str(sys.argv[1]) + "/latest"

    with open(file) as f: d = json.load(f)

    var ='Reference Evaluations'

    N  = len(d['Samples'])
    Ny = len(d['Samples'][0][var])
    Nx = len(d['Samples'][0]['Parameters']) - 0   #Parameters consist of one [Sigma] at the end!

    print("Number of samples = ", N)

    #load the variables into the matrices
    y = np.empty((N,Ny), dtype=np.float64)
    x = np.zeros((N,Nx))
    s = np.zeros((N,1))
    for k in range(N):
        y[k,:] = d['Samples'][k][var]
        x[k,:] = d['Samples'][k]['Parameters'][:Nx]
        #s[k,:] = d['Samples'][k]['Parameters'][Nx]

    #transform variables such that (mean = 0 and var = 1)
    transform = True
    if transform:
        scaler = preprocessing.StandardScaler().fit(y)
        y_scaled = scaler.transform(y)
        goal_scaled = scaler.transform(goal)
    else:
        y_scaled = y
        goal_scaled = goal


    dtype = [('index', int), ('error', float)]
    error = []

    for i in range(N):
        error.append((i, (goal_scaled[0,0] - y_scaled[i,0])**2 + (goal_scaled[0,1] - y_scaled[i,1])**2))
    error = np.array(error, dtype=dtype)
    error = np.sort(error, order='error')

    idx = np.empty(N, dtype=int)
    for i in range(N):
        idx[i] = error[i][0]

    y = y[idx]
    x = x[idx]

    y[:,1] *= 1

    y = np.around(y, decimals=4)
    x = np.around(x, decimals=4)

    print("[U_e, F] \t [eps, sigma, rc]\n")
    top = 20000
    t = 1
    i = 0
    y_old0 = y[0,0]
    y_old1 = y[0,1]


    data = np.empty((top,6))

    y[:,1] *= 1e-9

    print(t, ". ", y[0], " \t ", x[0])
    data[0,:] = np.concatenate((y[0], x[0]), axis=0)
    t += 1
    while(t <= top and i < N):
        if y_old0 != y[i,0] or y_old1 != y[i,1]:
            #print(t, ". ", y[i], " \t ", x[i])
            data[t-1,:] = np.concatenate((y[i], x[i]), axis=0)
            y_old0 = y[i,0]
            y_old1 = y[i,1]
            t += 1
        i += 1

    #print(data[:,2:])

    data = np.concatenate((y[:top], x[:top]), axis=1)
    print()
    print(np.mean(data, axis=0))
    print(np.std(data, axis=0))

    
