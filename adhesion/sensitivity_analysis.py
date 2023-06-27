#!/usr/bin/env python3
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numba
sys.path.append("model/")
from energy_force import *


#this code will check the given values how good the results are
@numba.jit(parallel=True)
def main():

    per_var = 0.1       #change of the value [e.g. 0.1 -> 10%]
    model = "lj"        #sw   or   sw2   or   lj

    if model == "sw":
        #parameters
        A_eps = 0.16
        B = 1.8
        sigma = 3.9
        rc = 10.1
        # A_eps = 0.05
        # B = 1.0
        # sigma = 3.4
        # rc = 10.0

        results = []
        for A_eps_ in [(1.0 - per_var)*A_eps, A_eps, (1.0 + per_var)*A_eps]:
            for B_ in [(1.0 - per_var)*B, B, (1.0 + per_var)*B]:
                for sigma_ in [sigma]:
                    for rc_ in [rc]:
                        Ue_F = energy_force(A_eps_ * 6.947695457055374e-21, B_, sigma_, rc_)
                        print("(", round(Ue_F[0],3), round(1e9*Ue_F[1],3), ")\t", round(A_eps_,3), round(B_,3), round(sigma_,3), round(rc_,3))

        print("\n")

        n = 40
        adhesion_data = np.empty((n,n,2), dtype=np.float64)
        
        A_eps_ = np.linspace(0.01, 1.0, n)
        B_     = np.linspace(0.5, 1.0, n)
        for i in range(n):
            for j in range(n):
                adhesion_data[i,j,:] = energy_force(A_eps_[j] * 6.947695457055374e-21, B_[i], sigma, rc)

        fig, axs = plt.subplots(2)

        cs1 = axs[0].contour(A_eps_, B_, adhesion_data[:,:,0], levels=[260])
        cs2 = axs[1].contour(A_eps_, B_, adhesion_data[:,:,1], levels=[0.0])

        #for i in range(len(cs1.collections)):print("hi ", cs1.collections[i].get_segments(), "\n")

        cs1.collections[0].set_linewidth(6)           # the red line
        cs1.collections[0].set_color('red')

        cs2.collections[0].set_linewidth(6)           # the red line
        cs2.collections[0].set_color('red')

        axs[0].set_title("Adhesion energy")
        axs[1].set_title("Total force (z-direction)")

        print(np.around(adhesion_data[::4,::4,0], decimals=1), "\n")
        print(np.around(1e9*adhesion_data[::4,::4,1], decimals=3))
        

        plt.xlabel("$A * \epsilon$")
        plt.ylabel("$B$")

        plt.show()

    elif model == "lj":
        #parameters
        eps = 0.063      #kcal/mol
        sigma = 2.73      #angstrom
        rc = 10.0


        for eps_ in [(1.0 - per_var)*eps, eps, (1.0 + per_var)*eps]:
            for sigma_ in [(1.0 - per_var)*sigma, sigma, (1.0 + per_var)*sigma]:
                for rc_ in [(1.0 - per_var)*rc, rc, (1.0 + per_var)*rc]:
                    Ue_F = energy_force_lj(eps_ * 6.947695457055374e-21, sigma_, rc_)
                    print("(", round(Ue_F[0],3), round(1e9*Ue_F[1],3), ")\t", round(eps_,3), round(sigma_,3), round(rc_,3))
        print("\n")

        n = 40
        adhesion_data = np.empty((n,n,2), dtype=np.float64)
        # eps_    = np.linspace((1.0 - 2*per_var)*eps, (1.0 + 2*per_var)*eps, n)
        # sigma_  = np.linspace((1.0 - per_var)*sigma, (1.0 + per_var)*sigma, n)
        #eps_   = np.linspace(0.01, 0.15, n)
        rc_    = np.linspace(5.0, 15.0, n)
        sigma_ = np.linspace(2.0, 4.0, n)
        for i in numba.prange(n):
            for j in range(n):
                adhesion_data[i,j,:] = energy_force_lj(eps * 6.947695457055374e-21, sigma_[i], rc_[j])

        fig, axs = plt.subplots(2)

        """
        sns.heatmap(adhesion_data[:,:,0], xticklabels=np.around(sigma_, decimals=3), yticklabels=np.around(eps_, decimals=3), ax=axs[0])
        sns.heatmap(adhesion_data[:,:,1], xticklabels=np.around(sigma_, decimals=3), yticklabels=np.around(eps_, decimals=3), ax=axs[1])
        """
        cs1 = axs[0].contour(rc_, sigma_, adhesion_data[:,:,0], levels=[200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320])
        cs2 = axs[1].contour(rc_, sigma_, adhesion_data[:,:,1], levels=[-1.2e-07, -1.0e-07, -8.0e-08, -6.0e-08, -4.0e-08, -2.0e-08,  0.0,  2.0e-08, 4.0e-08, 6.0e-08, 8.0e-08, 1.0e-07, 1.2e-07])

        cs1.collections[6].set_linewidth(6)           # the red line
        cs1.collections[6].set_color('red')

        cs2.collections[6].set_linewidth(6)           # the red line
        cs2.collections[6].set_color('red')

        axs[0].set_title("Adhesion energy")
        axs[1].set_title("Total force (z-direction)")

        print(np.around(adhesion_data[::4,::4,0], decimals=1), "\n")
        print(np.around(1e9*adhesion_data[::4,::4,1], decimals=3))
        

        plt.ylabel("$\sigma$")
        plt.xlabel("$r_c$")

        plt.show()


if __name__ == "__main__":
    main()