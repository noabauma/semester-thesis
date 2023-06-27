import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("model/")
from energy_force import *

"""
def sw2_pot(r, A_eps, B, sigma, rc):
    return A_eps*(B*((sigma/r)**4) - 1.0)*np.exp(sigma/(r-rc))

def sw2_for(r_ij_z, r2, r, A_eps, B, sigma, rc):
    
    rs2 = (sigma*sigma) / r2
    B_rs4 = B * rs2 * rs2
    r_rc = r - rc
    exp = np.exp(sigma / r_rc)
    A_eps_exp = A_eps * exp
    phi = (sigma * (B_rs4 - 1.0))/(r_rc*r_rc*r) + (4.0*B_rs4)/r2

    return phi * A_eps_exp * r_ij_z    #I only want the z direction

def lj_pot(r, eps, sigma, rc):
    sr  = sigma/r
    sr2 = sr*sr
    sr4 = sr2*sr2
    sr6 = sr4*sr2

    return 4.0*eps*(sr6*sr6 - sr6)

def lj_for(r_ij_z, r2, r, eps, sigma, rc):
    sr  = sigma/r
    sr2 = sr*sr
    sr4 = sr2*sr2
    sr6 = sr4*sr2

    return 24.0*eps*(2.0*sr6*sr6 - sr6)*(r_ij_z/r2)
"""

def main():
    #sw2 parameters
    sw2_par =  np.array([[ 0.0369,  1.4395,  3.09,   10.3375,  0.0996],
                [ 0.0279,  0.7778,  3.6056, 11.5894,  0.0983],
                [ 0.0584,  0.858 ,   3.5269,  9.5546,  0.0983],
                [ 0.0721,  0.8705,  3.4979,  9.0845,  0.0973],
                [ 0.0271,  0.6039,  3.8264,  11.8512,  0.0989],
                [ 0.0507,  1.4735,  3.0739,  9.5385,  0.0986],
                [ 0.0368,  1.1352,  3.271,  10.4849,  0.0989],
                [ 0.1146,  0.9931,  3.4076,  8.2034,  0.0996],
                [ 0.0835,  0.2679,  4.7218,  9.5836,  0.0981],
                [ 0.0163,  1.6079,  2.9789,  12.9289,  0.0988]])


    #lj parameters
    eps = 0.063     #kcal/mol
    sigma_lj = 3.4
    rc_lj = 10.0


    #Potential
    if False:
        r = np.linspace(3.0, 8.2, 100)

        for i in range(sw2_par.shape[0]):
            #if i == 2 or i == 7 or i == 5 or i == 8:
                
            sw2_r = sw2_pot(r, sw2_par[i,0], sw2_par[i,1], sw2_par[i,2], sw2_par[i,3])
            plt.plot(r, sw2_r, label=1+i)
        
        lj_r = lj_pot(r, eps, sigma_lj, rc_lj)
        plt.plot(r, lj_r, "r--", label="LJ potential")

        plt.ylim((-0.075, 0.05))

        plt.title("Top 10 SW2 Potentials vs LJ Potential")
        plt.ylabel("V(r) $[kcal/mol]$")
        plt.xlabel("r $[\AA]$")
        plt.legend()
        plt.show()

    #normal Force
    else:
        r = np.linspace(3.0, 8.2, 100)
        """
        for i in range(sw2_par.shape[0]):

            U_e, F = energy_force(sw2_par[i,0] * 6.947695457055374e-21, sw2_par[i,1], sw2_par[i,2], sw2_par[i,3])
            print(i+1, "[", U_e, ", ", F, "]")
            if i != 11:
                sw2_r = sw2_for(r, r*r, r, sw2_par[i,0], sw2_par[i,1], sw2_par[i,2], sw2_par[i,3])
                plt.plot(r, -sw2_r, label=str(i+1) + ". force")

                sw2_r = sw2_pot(r, sw2_par[i,0], sw2_par[i,1], sw2_par[i,2], sw2_par[i,3])
                plt.plot(r, sw2_r, label=str(i+1) + ". potential")
                break
        """

        U_e, F = energy_force_lj(eps * 6.947695457055374e-21, sigma_lj, rc_lj)
        print("lj: [", U_e, ", ", F, "]")

        lj_r = lj_for(r, r*r, r, eps, sigma_lj, rc_lj)
        plt.plot(r, -lj_r, "r--", label="LJ force")

        lj_r = lj_pot(r, eps, sigma_lj, rc_lj)
        plt.plot(r, lj_r, "m--", label="LJ potential")

        plt.ylim((-0.075, 0.05))

        plt.title("Top 10 SW2 Forces vs LJ Force")
        plt.ylabel("$F_N(r) [nN]$")
        plt.xlabel("r $[\AA]$")
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    main()