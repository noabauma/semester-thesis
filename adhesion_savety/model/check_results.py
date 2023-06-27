from energy_force import *

#Parameters
A = 1.0
B = 1.0
epsilon = 1.0
sigma = 1.0
rc = 1.0

U_e, F = energy_force(A, B, epsilon, sigma, rc)
print("adhesion energy [mJ/m^2] = ", U_e, "\ntotal force in z-direction [N] = ", F)