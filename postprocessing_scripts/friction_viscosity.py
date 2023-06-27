import numpy as np
import pint
import os
import matplotlib.pyplot as plt

#This python script will calculate the friction coef. and viscosity of an equilibrium (flow-free) CNT filled with water
#via the Green-Kubo method



def main():
    ureg = pint.UnitRegistry()

    #parameters
    kB = 1.38064852e-23 * ureg.kg * (ureg.m/ureg.s)**2 / ureg.K     #Boltzmann constant            [kg * (m/s)^2 * K^-1]
    T  = 300.0 * ureg.K                                             #Temperature in the simulation [K]

    R = 75.392*0.5 * ureg.angstrom  #Radius of the CNT
    L = 200.0      * ureg.angstrom  #Length of the CNT
    V = np.pi*R*R*L                 #Volume of the domain or CNT (or non if formula (13) is taken)?
    A = 2.0*np.pi*R*L               #Surface inside the CNT
    V = ureg('10.0 nm')**3

    print("Loading P", flush=True)
    P = np.loadtxt("../other/h5_argon/stress_tensor.csv", dtype=np.float64, comments="t", delimiter=",") #t, xy, yx, yz, zy, zx, xz
    t = P[:, 0]  * ureg.fs
    P = P[:, 1:] * ureg('kg angstrom^2 / fs^2')
    """
    print("Loading P_cp", flush=True)
    P_cp = np.loadtxt("../h5_argon/stress_tensor_cp.csv", dtype=np.float64, comments="t", delimiter=",") #t, xy, yx, yz, zy, zx, xz
    P_cp = P_cp[:, 1:] * ureg('kg angstrom^2 / fs^2')
    print(P.shape, P_cp.shape, flush=True)
    P -= P_cp   #remove c-w stress tensor
    """
    print("Loading P coord", flush=True)
    coord = np.loadtxt("../other/h5_argon/stress_tensor.csv", dtype=str, delimiter=",", max_rows=1)
    coord = coord[1:]
    print(coord)
    print("Loading F", flush=True)
    F = np.loadtxt("../h5_w_in_cnt111/tangential_force.csv", dtype=np.float64, comments="t", delimiter=",") #t, x, y, z
    F = F[:, 1:] * ureg('kg angstrom / fs^2')
    print("Done", flush=True)

    K = P.shape[1]   #xy, yx, yz, zy, zx, xz

    assert P.shape[0] == F.shape[0]
    
    ensemble_gaps = 1000                              
    m = P.shape[0]
    n_ensembles = m // ensemble_gaps

    print("#ensembles =\t",  n_ensembles, "\n")

    dt = t[1] - t[0]            #Step sizes: [0fs, 2fs, 4fs, 6fs, 8fs, ...]

    dt = ureg('2 fs')
    print(dt)

    P *= 1.0 / V                #Because of the formulas

    Pen = P[:n_ensembles * ensemble_gaps].reshape(n_ensembles, ensemble_gaps, K)
    G = np.mean(np.cumsum(Pen, axis=1) ** 2, axis=0)
    assert G.shape == (ensemble_gaps, K)

    P0 = Pen[:, 0, :]
    P_GK = np.cumsum(P0 * np.moveaxis(Pen, 1, 0), axis=0)
    P_GK = np.mean(P_GK, axis=1)
    assert P_GK.shape == (ensemble_gaps, K)

    Fzen = F[:n_ensembles * ensemble_gaps, 2].reshape(n_ensembles, ensemble_gaps)
    Fz0 = Fzen[:, 0]
    F_ACF_all = np.cumsum(Fz0 * Fzen.T, axis=0)
    F_ACF = np.mean(F_ACF_all, axis=1)

    #constant fit for F ACF integral from 4ps to 5ps
    F_mean = np.mean(F_ACF[800:])
    

    P_GK_tot = np.zeros((ensemble_gaps))
    G_mean = np.zeros((ensemble_gaps))
    cnt = 0
    for k in range(K):
        #if coord[k] == 'Pxz' or coord[k] == 'Pyz' or coord[k] == 'Pzx' or coord[k] == 'Pzy':
        if coord[k] != 'Pxx' and coord[k] != 'Pyy' and coord[k] != 'Pzz':
        #if coord[k] == 'Pxz' or coord[k] == 'Pyz':
            P_GK_tot += P_GK[:,k]
            G_mean += G[:,k]
            cnt += 1
    G_mean /= cnt
    P_GK_tot /= cnt

    P_GK_mean = np.mean(P_GK_tot[100:])


    eta     = (G_mean*dt*dt*V)/(2.0*kB*T)
    eta_GK  = P_GK_mean*dt*V/(kB*T)
    lambda_ = F_mean*dt/(A*kB*T)

    #linear fit to find slope
    x = (t[:ensemble_gaps] - t[0])
    p = np.polyfit(x.m_as(ureg.s)[100:], eta.m_as(ureg.mPa*ureg.s**2)[100:], deg=1)

    print("viscosity (Einstein) = ", p[0], "\t\t [mPa*s]")
    print("viscosity (Green-Kubo) = ", eta_GK.to('mPa s'))
    print("friction coefficient = ", lambda_.to(ureg.N * ureg.s / (ureg.m**3)), "\t UNIT equals to: [Ns/m^3] or [kg/(s*m^2)]")

    


    
    #plot for debugging
    x = x.m_as('ps')
    #viscosity Einstein
    G *= (dt*dt*V)/(2.0*kB*T)

    G = G.m_as('mPa s^2')

    """
    for k in range(K):
        if coord[k] != 'Pxx' and coord[k] != 'Pyy' and coord[k] != 'Pzz':
            plt.plot(x, G[:,k], label=coord[k])

    G_mean *= (dt*dt*V)/(2.0*kB*T)
    G_mean = G_mean.m_as('mPa s^2')
    plt.plot(x, G_mean, label='average')
    p = np.polyfit(x[100:], G_mean[100:], deg=1)
    plt.plot(x[100:], (p[0]*x + p[1])[100:], 'r--', label='linear fit')
    plt.ylabel(r'$\mu(t) t\ [mPa\ s^2]$')
    plt.xlabel(r'$t\ [ps]$')
    plt.grid()
    plt.legend()
    plt.show()

    
    #viscosity Green Kubo
    P_GK *= dt*V/(kB*T)
    P_GK_tot *= dt*V/(kB*T)
    for k in range(K):
        if coord[k] != 'Pxx' and coord[k] != 'Pyy' and coord[k] != 'Pzz':
            plt.plot(x, P_GK[:,k].m_as('mPa s'), label=coord[k])

    plt.plot(x, P_GK_tot.m_as('mPa s'), label='average')
    plt.plot(x[100:], np.full(x[100:].shape, np.mean(P_GK_tot[100:].m_as('mPa s'))), 'r--', label='constant fit')
    plt.ylabel(r'$\mu\ [mPa\ s]$')
    plt.xlabel(r'$t\ [ps]$')
    plt.grid()
    plt.legend()
    plt.show()
    """
    
    #friction coef plot (for debugging)
    F_ACF     *= dt/(A*kB*T)
    F_ACF_all *= dt/(A*kB*T)

    #for i in range(215, 220): plt.plot(x, F_ACF_all[:,i].m_as('N s m^-3'))
    #plt.plot(x, F_ACF.m_as('N s m^-3'), label='average')
    plt.errorbar(x, F_ACF.m_as('N s m^-3'), 7e-7*np.var(F_ACF_all.m_as('N s m^-3'), axis=1), label='average', errorevery=30)
    plt.plot(x[800:], np.full(x[800:].shape, np.mean(F_ACF[800:].m_as('N s m^-3'))), 'r--', label='constant fit')
    
    #plt.title("auto correlation function of the total tangential force")
    plt.ylabel(r'$\lambda\ [\frac{N s}{m^3}]$')
    plt.xlabel(r'$t\ [ps]$')
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()
    
    

if __name__ == "__main__":
    main()
