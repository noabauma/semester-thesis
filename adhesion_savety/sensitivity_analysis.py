#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append("model/")
from energy_force import *


#this code will check the given values how good the results are
if __name__ == "__main__":

    per_var = 0.1       #change of the value [e.g. 0.1 -> 10%]
    model = "lj"        #sw   or   sw2   or   lj

    if model == "sw2":
        #parameters
        A_eps = 50.0
        B = 0.5
        sigma = 3.5
        rc = 7.25

        results = []
        for A_eps_ in [(1.0 - per_var)*A_eps, A_eps, (1.0 + per_var)*A_eps]:
            for B_ in [(1.0 - per_var)*B, B, (1.0 + per_var)*B]:
                for sigma_ in [(1.0 - per_var)*sigma, sigma, (1.0 + per_var)*sigma]:
                    for rc_ in [(1.0 - per_var)*rc, rc, (1.0 + per_var)*rc]:
                        #results.append(energy_force(A_, B_, epsilon_, sigma_, rc_))
                        print(energy_force(A_eps_ * 6.947695457055374e-21, B_, sigma_, rc_), end=" ")
        print("\n")

    elif model == "sw":
        #parameters
        A_eps = 50.0
        B = 0.5
        sigma = 3.5
        rc = 7.25

        results = []
        for A_eps_ in [(1.0 - per_var)*A_eps, A_eps, (1.0 + per_var)*A_eps]:
            for B_ in [(1.0 - per_var)*B, B, (1.0 + per_var)*B]:
                for sigma_ in [(1.0 - per_var)*sigma, sigma, (1.0 + per_var)*sigma]:
                    for rc_ in [(1.0 - per_var)*rc, rc, (1.0 + per_var)*rc]:
                        #results.append(energy_force(A_, B_, epsilon_, sigma_, rc_))
                        print(energy_force(A_eps_ * 6.947695457055374e-21, B_, sigma_, rc_), end=" ")
        print("\n")

    elif model == "lj":
        #parameters
        eps = 0.105      #kcal/mol
        sigma = 3.851      #angstrom
        rc = 10.0
        # eps = 3.750e-01      #kcal/mol
        # sigma = 2.692e+00      #angstrom
        # rc = 1.319e+01


        for eps_ in [(1.0 - per_var)*eps, eps, (1.0 + per_var)*eps]:
            for sigma_ in [(1.0 - per_var)*sigma, sigma, (1.0 + per_var)*sigma]:
                for rc_ in [(1.0 - per_var)*rc, rc, (1.0 + per_var)*rc]:
                    print(energy_force_lj(eps_ * 6.947695457055374e-21, sigma_, rc_), eps_, sigma_, rc_)
        print("\n")

        n = 10
        adhesion_energy = np.empty((n,n), dtype=np.float64)
        eps_    = np.linspace((1.0 - per_var)*eps, (1.0 + per_var)*eps, n)
        sigma_  = np.linspace((1.0 - per_var)*sigma, (1.0 + per_var)*sigma, n)
        for i in range(n):
            for j in range(n):
                adhesion_energy[i,j] = energy_force_lj(eps_[i] * 6.947695457055374e-21, sigma_[j], rc)[0]

        ax = sns.heatmap(adhesion_energy, xticklabels=np.around(sigma_, decimals=3), yticklabels=np.around(eps_, decimals=3))


        plt.xlabel("sigma")
        plt.ylabel("epsilon")

        plt.show()
