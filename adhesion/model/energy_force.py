import numpy as np
import numba



graphene_sheets = np.loadtxt("model/grs_2_50_50.txt", dtype=np.float64)
#graphene_sheets = np.loadtxt("model/grs_2_200_200.txt", dtype=np.float64)

layer_distance = np.max(graphene_sheets[:,2]) - np.min(graphene_sheets[:,2])
Lx = np.max(graphene_sheets[:,0]) - np.min(graphene_sheets[:,0])
Ly = np.max(graphene_sheets[:,1]) - np.min(graphene_sheets[:,1])
per_area = 1.0/(Lx*Ly*1e-20)  #[m^-2]
n = graphene_sheets.shape[0]
n_half = int(n*0.5)

graphene1 = graphene_sheets[:n_half,:]
graphene2 = graphene_sheets[n_half:,:]
#graphene2[:,2] += 0.6
d_ = abs(graphene2[0,2] - graphene1[0,2])

print("Graphene layer distance in A: ", graphene1[0,2], graphene2[0,2])


@numba.njit
def l2_distance_sq(atom1, atom2):
    dx = atom2[0] - atom1[0]
    dy = atom2[1] - atom1[1]
    dz = atom2[2] - atom1[2]

    r2 = (dx * dx + dy * dy + dz * dz)

    return r2


@numba.njit
def sw2_pot(r, A_eps, B, sigma, rc):
    return A_eps*(B*((sigma/r)**4) - 1.0)*np.exp(sigma/(r-rc))

@numba.njit
def sw2_for(r_ij_z, r2, r, A_eps, B, sigma, rc):
    
    rs2 = (sigma*sigma) / r2
    B_rs4 = B * rs2 * rs2
    r_rc = r - rc
    exp = np.exp(sigma / r_rc)
    A_eps_exp = A_eps * exp
    phi = (sigma * (B_rs4 - 1.0))/(r_rc*r_rc*r) + (4.0*B_rs4)/r2

    return phi * A_eps_exp * r_ij_z    #I only want the z direction


@numba.njit(parallel=False)
def energy_force(A_eps, B, sigma, rc):
    rc2 = rc*rc
    energy = 0.0
    force = 0.0

    # numba.prange requires parallel=True flag to compile.
    # It causes the loop to run in parallel in multiple threads.
    for i in numba.prange(n_half):
        for j in range(n_half):
            r2 = l2_distance_sq(graphene1[i], graphene2[j])
            if r2 > rc2:
                continue

            r = np.sqrt(r2)
            energy += sw2_pot(r, A_eps, B, sigma, rc)
            force += sw2_for(graphene1[i,2] - graphene2[j,2], r2, r, A_eps, B, sigma, rc)

    return (-energy*per_area*1e3, force*1e10)



#these are the functions for "run_tmcmc2.py" multiplied parameters
@numba.njit
def sw2_pot2(r, C, D, sigma, rc):
    return (C/(r*r*r*r) - D)*np.exp(sigma/(r-rc))

@numba.njit
def sw2_for2(r_ij_z, r2, r, C, D, sigma, rc):
    
    r4_inv = 1.0/(r2*r2)
    r_rc   = r - rc
    exp    = np.exp(sigma / r_rc)
    lhs1   = sigma*(C*r4_inv - D)/(r_rc*r_rc*r)
    lhs2   = 4.0*C*r4_inv/r2
    return (lhs1 + lhs2) * exp * r_ij_z


@numba.njit(parallel=False)
def energy_force2(C, D, sigma, rc):
    rc2 = rc*rc
    energy = 0.0
    force = 0.0

    for i in numba.prange(n_half):
        for j in range(n_half):
            r2 = l2_distance_sq(graphene1[i], graphene2[j])
            if r2 > rc2:
                continue

            r = np.sqrt(r2)
            energy += sw2_pot2(r, C, D, sigma, rc)
            force += sw2_for2(graphene1[i,2] - graphene2[j,2], r2, r, C, D, sigma, rc)

    return (-energy*per_area*1e3, force*1e10)


#these are the functions for "run_tmcmc2.py" multiplied parameters
@numba.njit
def lj_pot(r, eps, sigma, rc):
    sr  = sigma/r
    sr2 = sr*sr
    sr4 = sr2*sr2
    sr6 = sr4*sr2

    return 4.0*eps*(sr6*sr6 - sr6)

@numba.njit
def lj_for(r_ij_z, r2, r, eps, sigma, rc):
    sr  = sigma/r
    sr2 = sr*sr
    sr4 = sr2*sr2
    sr6 = sr4*sr2

    return 24.0*eps*(2.0*sr6*sr6 - sr6)*(r_ij_z/r2)



@numba.njit(parallel=False)
def energy_force_lj(eps, sigma, rc):
    rc2 = rc*rc
    energy = 0.0
    force = 0.0

    for i in numba.prange(n_half):
        for j in range(n_half):
            r2 = l2_distance_sq(graphene1[i], graphene2[j])
            if r2 > rc2:
                continue

            r = np.sqrt(r2)
            energy += lj_pot(r, eps, sigma, rc)
            force += lj_for(graphene1[i,2] - graphene2[j,2], r2, r, eps, sigma, rc)

    return (-energy*per_area*1e3, force*1e10)