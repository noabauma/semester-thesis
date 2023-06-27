import numpy as np
import pint
import os
import matplotlib.pyplot as plt

#This python script will calculate the friction coef. and viscosity of an equilibrium (flow-free) CNT filled with water
#via the Green-Kubo method



if __name__ == "__main__":
    ureg = pint.UnitRegistry()

    #parameters
    kB = 1.38064852e-23 * ureg.kg * (ureg.m/ureg.s)**2 / ureg.K     #Boltzmann constant            [kg * (m/s)^2 * K^-1]
    T  = 300.0 * ureg.K                                             #Temperature in the simulation [K]

    R = 75.392*0.5 * ureg.angstrom  #Radius of the CNT
    L = 200        * ureg.angstrom  #Length of the CNT
    V = np.pi*R*R*L                 #Volume of the domain or CNT (or non if formula (13) is taken)?
    A = 2.0*np.pi*R*L               #Surface inside the CNT

    P = np.loadtxt("../h5_w_in_cnt4/stress_tensor.csv"   , dtype=np.float64, comments="t", delimiter=",") #t, xy, yx, yz, zy, zx, xz
    F = np.loadtxt("../h5_w_in_cnt4/tangential_force.csv", dtype=np.float64, comments="t", delimiter=",") #t, x, y, z

    K = 6   #xy, yx, yz, zy, zx, xz

    m  = P.shape[0]                 #considering both files being the same size!
    m_ = F.shape[0]
    if m != m_:
        os.exit(1)
    
    print("#outputs   =\t", m)

    ensemble_gaps = 2500                              
    n_ensembles = int(m/ensemble_gaps)

    print("#ensembles =\t",  n_ensembles, "\n")

    dt = (P[1,0] - P[0,0]) * ureg.fs            #Step sizes: [0fs, 2fs, 4fs, 6fs, 8fs, ...]

    P_ACF = np.zeros((ensemble_gaps, K), dtype=np.float64)  
    F_ACF = np.zeros((ensemble_gaps), dtype=np.float64)
    F_ACF_all = np.zeros((m), dtype=np.float64)
    P_GK  = np.zeros((ensemble_gaps, K), dtype=np.float64)

    V_inv = 1.0/V.magnitude
    P[:,1:] *= V_inv
    
    for i in range(n_ensembles):
        F_0 = F[i*ensemble_gaps, 3]  #z coordinate
        P_0 = P[i*ensemble_gaps,1:]

        ensemble_F = 0.0
        ensemble_P = np.zeros((K), dtype=np.float64)
        for j in range(ensemble_gaps):

            for k in range(K):
                P_ACF[j,k] += (np.sum(P[i*ensemble_gaps:(i*ensemble_gaps + j), k+1]))**2

            ensemble_P += P_0*P[i*ensemble_gaps + j, 1:]
            P_GK[j,:]  += ensemble_P

            ensemble_F += F_0*F[i*ensemble_gaps + j, 3]
            F_ACF[j] += ensemble_F
            F_ACF_all[i*ensemble_gaps + j] = ensemble_F


    #average over ensembles
    P_ACF /= n_ensembles
    F_ACF /= n_ensembles
    P_GK  /= n_ensembles

    #constant fit for F ACF integral from 4ps to 5ps
    F_tot = np.mean(F_ACF[2000:])

    #insert pint units
    P_ACF *= (ureg.kg * (ureg.angstrom / ureg.fs)**2 / ureg.angstrom**3)**2 
    F_tot *= (ureg.kg * ureg.angstrom / (ureg.fs**2))**2
    P_GK  *= (ureg.kg * (ureg.angstrom / ureg.fs)**2 / ureg.angstrom**3)**2

    eta     = ((P_ACF[:,2] + P_ACF[:,4])*dt*dt*V)/(2.0*kB*T)    #yz & xz
    lambda_ = F_tot*dt/(A*kB*T)

    #linear fit to find slope
    x_ = ((P[:ensemble_gaps, 0] - P[0,0]) * ureg.fs).to(ureg.s).magnitude
    p = np.polyfit(x_[1000:], eta.to(ureg.Pa*ureg.s**2).magnitude[1000:], deg=1)

    print("viscosity (Einstein) = ", p[0], "\t\t UNIT equals to: [Pa*s] or [Ns/m^2] or [kg/(s*m)]")
    print("friction coefficient = ", lambda_.to(ureg.N * ureg.s / (ureg.m**3)), "\t UNIT equals to: [Ns/m^3] or [kg/(s*m^2)]")


    
    #plot for debugging
    x = (P[:ensemble_gaps, 0] - P[0,0])/1000.0  #[ps]
    #viscosity Einstein
    P_ACF *= dt*dt

    coord = ['xx','xy','xz','yy','yz','zz']
    for k in range(K):
        if coord[k] == 'xz' or coord[k] == 'yz':
            plt.plot(x, P_ACF[:,k].to(ureg.Pa**2 *ureg.s**2).magnitude, label=coord[k])

    plt.plot(x, ((P_ACF[:,2] + P_ACF[:,4])*0.5).to(ureg.Pa**2 * ureg.s**2).magnitude, label='average')
    p = np.polyfit(x[1000:], ((P_ACF[:,2] + P_ACF[:,4])*0.5).to(ureg.Pa**2 * ureg.s**2).magnitude[1000:], deg=1)
    plt.plot(x[1000:], (p[0]*x + p[1])[1000:], 'r--', label='linear fit')
    plt.ylabel(r'$\lambda [Pa\ s^2]$')
    plt.xlabel(r'$t\ [ps]$')
    plt.grid()
    plt.legend()
    plt.show()


    #viscosity Green Kubo
    P_GK_tot = np.zeros((ensemble_gaps), dtype=np.float64)
    P_GK *= dt*V/(1.0*kB*T)
    for k in range(K):
        if coord[k] == 'xz' or coord[k] == 'yz':
            plt.plot(x, P_GK[:,k].to(ureg.Pa*ureg.s).magnitude, label=coord[k])
            P_GK_tot += P_GK[:,k]

    P_GK_tot /= 2.0

    plt.plot(x, P_GK_tot.to(ureg.Pa*ureg.s).magnitude, label='average')
    plt.ylabel(r'$\lambda [Pa\ s]$')
    plt.xlabel(r'$t\ [ps]$')
    plt.grid()
    plt.legend()
    plt.show()

    #friction coef plot (for debugging)
    
    F_ACF *= (ureg.kg * ureg.angstrom / (ureg.fs**2))**2
    F_ACF *= dt

    F_ACF_all *= (ureg.kg * ureg.angstrom / (ureg.fs**2))**2
    F_ACF_all *= dt
    
    for i in range(15, 20):
        plt.plot(x, F_ACF_all[i*ensemble_gaps:(i+1)*ensemble_gaps].to(ureg.N**2 * ureg.s).magnitude)
    plt.plot(x, F_ACF.to(ureg.N**2 * ureg.s).magnitude, label='average')
    
    #plt.title("auto correlation function of the total tangential force")
    plt.ylabel(r'$\int_{0}^{T} \langle F_0 F(t) \rangle\ \,dt\ [N^2s]$')
    plt.xlabel(r'$t\ [ps]$')
    plt.grid()
    plt.legend()
    plt.show()
    