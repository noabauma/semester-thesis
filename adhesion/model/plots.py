#! /usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt

#create a boxplot of the Reference Evaluations
def box_plot(file):

    with open(file) as f: d = json.load(f)

    var ='Reference Evaluations'

    N  = len(d['Samples'])
    Ny = len(d['Samples'][0][var])
    #Nx = len(d['Samples'][0]['Parameters']) - 1   #Parameters consist of one [Sigma] at the end!

    print("Number of samples = ", N)

    y = np.empty((N,Ny), dtype=np.float64)
    #x = np.zeros((N,Nx))
    #s = np.zeros((N,1))
    for k in range(N):
        y[k,:] = d['Samples'][k][var]
        #x[k,:] = d['Samples'][k]['Parameters'][:Nx]
        #s[k,:] = d['Samples'][k]['Parameters'][Nx]

    fig, axs = plt.subplots(1,2)

    axs[0].boxplot(y[:,0])
    axs[0].axhline(260.0, color='r', linestyle='--')
    axs[1].boxplot(y[:,1])
    axs[1].axhline(0.0, color='r', linestyle='--')

    axs[0].title.set_text("Adhesion Energy")
    axs[1].title.set_text("Total force (z-direction)")

    plt.show()



#box plot only for adhesion
def box_plot4(file):

    with open(file) as f: d = json.load(f)

    var ='Reference Evaluations'

    N  = len(d['Samples'])
    Ny = len(d['Samples'][0][var])
    #Nx = len(d['Samples'][0]['Parameters']) - 1   #Parameters consist of one [Sigma] at the end!

    print("Number of samples = ", N)

    y = np.empty((N,Ny), dtype=np.float64)
    #x = np.zeros((N,Nx))
    #s = np.zeros((N,1))
    for k in range(N):
        y[k,:] = d['Samples'][k][var]
        #x[k,:] = d['Samples'][k]['Parameters'][:Nx]
        #s[k,:] = d['Samples'][k]['Parameters'][Nx]

    fig, axs = plt.subplots(1,1)

    axs.boxplot(y)
    axs.axhline(260.0, color='r', linestyle='--')

    axs.title.set_text("Adhesion Energy")

    plt.show()

#here for debugging
#box_plot('../_korali_result_propagation/latest')