#!/usr/bin/env python

#from wca_simulations import *
#from visc_frct_simulation import *
from mirheo_scripts import *

if __name__ == "__main__":
    rc      = 10.0                  #[A]
    eps     = 365.1/6.02214076e23   #[J]
    sigma   = 3.19                  #[A]
    A       = 0.25                  #[1]    (7.049556277)
    B       = 0.6022245584          #[1]

    A_eps = A*eps

    wca_grs = droplet_grs(A_eps, B, sigma, rc)

    print("WCA on graphene = ", wca_grs)

    wca_cnt = droplet_cnt(A_eps, B, sigma, rc)

    print("WCA in cnt = ", wca_cnt)

    eta, lambda_ = water_in_cnt(A_eps * 6.947695457055374e-21, B, sigma, rc)

    print("Viscosity = ", eta, "\nFriction coef = ", lambda_)