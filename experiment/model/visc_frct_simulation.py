import mirheo as mir
import pint
import numpy as np
import h5py


#simulating water inside CNT and calculate the friction coef & viscosity
ureg = pint.UnitRegistry()

# Define Mirheo's unit system.
ureg.define('mirL = 1 angstrom')
ureg.define('mirT = 1 fs')
ureg.define('mirM = 1 kg')
mir.set_unit_registry(ureg)

#load particles for Water in CNT
water = h5py.File('particles/water_for_cnt_96_0_200/pv.PV-00000.h5', 'r')
w_pos = water["position"][()]
w_vel = water["velocity"][()]

c_pos = h5py.File('particles/h5_cnt_96_0_200/pv_c-00001.h5', 'r')["position"][()]
Ncnt = c_pos.shape[0]
c_vel = np.zeros((Ncnt, 3), dtype=np.float64)

#determine the space for periodic boundary
ac = False                                                                  #armchair or zigzag pattern in horizontal layer?
L_cnt = np.max(c_pos[:,2]) - np.min(c_pos[:,2])                                #length of the cnt [angstrom]
L = (L + 1.23353) * ureg.angstrom if ac else (L + 1.42436) * ureg.angstrom  # 1.23353 = 1.42436*cos(30°)


#shift particles
shift = np.array([100.0 - np.mean(c_pos[:,0]), 100.0 - np.mean(c_pos[:,1]), 0.1 - np.min(c_pos[:,2])])
c_pos += shift
shift = np.array([100.0 - np.mean(w_pos[:,0]), 100.0 - np.mean(w_pos[:,1]), 0.1 - np.min(w_pos[:,2])])
w_pos += shift


dt = ureg('2 fs')            		   # timestep   (petros 2003)
rc_ww = ureg('0.432 nm')

ranks  = (1, 1, 1)
domain = (ureg('200.0 angstrom'), ureg('200.0 angstrom'), L)



#Friction ceof & viscosity parameters
#parameters
kB = 1.38064852e-23 * ureg.kg * (ureg.m/ureg.s)**2 / ureg.K     #Boltzmann constant            [kg * (m/s)^2 * K^-1]
T  = 300.0 * ureg.K                                             #Temperature in the simulation [K]

R = 75.392*0.5 * ureg.angstrom  #Radius of the CNT
L = 200        * ureg.angstrom  #Length of the CNT
V = np.pi*R*R*L                 #Volume of the domain or CNT (or non if formula (13) is taken)?
A = 2.0*np.pi*R*L               #Surface inside the CNT

V3_kB_T_inv = 1.0/(3.0*V*kB*T)
A_kB_T_inv = 1.0/(A*kB*T)



def water_in_cnt(A_eps, B, sigma, rc_cw):
    ##############################
    # Stage 0. Initialize Mirheo
    ##############################
    u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log')
    
    #register particles
    pv_w = mir.ParticleVectors.ParticleVector('pv_w', mass=ureg('18.015/6.02214076e23 g'))
    ic = mir.InitialConditions.FromArray( pos=w_pos, vel=w_vel)
    u.registerParticleVector(pv_w, ic)

    pv_c = mir.ParticleVectors.ParticleVector('pv_c', mass=ureg('12.0107/6.02214076e23 g'))
    ic = mir.InitialConditions.FromArray( pos=c_pos, vel=c_vel)
    u.registerParticleVector(pv_c, ic)


    ###water-water Interactions###
    sw3_ww = mir.Interactions.Triplewise('sw3_ww', rc_ww, kind='SW3', lambda_=23.15, epsilon=ureg('6.189/6.02214076e23 kcal'), theta=1.9106332362490186, gamma=1.2, sigma=ureg('0.23925 nm'))
    sw2_ww = mir.Interactions.Pairwise('sw2_ww', rc_ww, kind='SW', epsilon=ureg('6.189/6.02214076e23 kcal'), sigma=ureg('0.23925 nm'), A=7.049556277, B=0.6022245584)
    u.registerInteraction(sw3_ww)
    u.registerInteraction(sw2_ww)
    u.setInteraction(sw3_ww, pv_w, pv_w, pv_w)
    u.setInteraction(sw2_ww, pv_w, pv_w)
    

    ###carbon-water Interactions###
    sw2_cw = mir.Interactions.Pairwise('sw2_cw', rc_cw, kind='SW', epsilon=A_eps*ureg.J, sigma=sigma*ureg.angstrom, A=1.0, B=B)    #Koralis values
    u.registerInteraction(sw2_cw)
    u.setInteraction(sw2_cw, pv_c, pv_w)
    


    ##############################
    # Stage 1. stabilize with Minimize
    ##############################
    minimize = mir.Integrators.Minimize('int_minimize', max_displacement=ureg('1.0 angstrom'))
    u.registerIntegrator(minimize)
    u.setIntegrator(minimize, pv_w)

    u.run(10002, dt=dt)

    u.deregisterIntegrator(minimize)
    del minimize

    ##############################
    # Stage 2. Velocity-Verlet & get temperature right (equilibrium) (should take 0.2ns)
    ##############################
    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv_w)
    

    ####Plugins####				
    u.registerPlugins(mir.Plugins.createBerendsenThermostat('thermostat', [pv_w], T=ureg('300.0 K'), tau=ureg('2.0 ps')))   #(Werder et al. 2003)
    
    u.run(100002, dt=dt)

    ##############################
    # Stage 3. Save stress tensor & total tangential force with Miheo plugins into csv files
    ##############################
    path = "data/h5_w_in_cnt/"
    dump_every = 1

    ####Viscosity####
    u.registerPlugins(mir.Plugins.createStressTensor('stress_tensor', pv_w, dump_every, path))

    ####Friction coef.####
    u.registerPlugins(mir.Plugins.createTotalForceSaver('tangential_force', pv_w, dump_every, path))

    u.run(1000002, dt=dt)


    ##############################
    # Stage 4. Calculate friction coefficient & viscosity
    ##############################
    if u.isComputeTask():
        P = np.loadtxt("data/h5_w_in_cnt/stress_tensor.csv"   , dtype=np.float64, comments="t", delimiter=",")
        F = np.loadtxt("data/h5_w_in_cnt/tangential_force.csv", dtype=np.float64, comments="t", delimiter=",")

        m  = P.shape[0]                 #considering both files being the same size!

        ensemble_gaps = 16                              #Step sizes atleast: [0fs, 2fs, 4fs, 6fs, 8fs, ...]
        n_ensembles = int(m/ensemble_gaps)

        dt_ = (P[ensemble_gaps-1,0] - P[0,0]) * ureg.fs

        ensemble_avg_P = 0.0
        ensemble_avg_F = 0.0
        for i in range(n_ensembles):
            Pxy_0 = P[i*ensemble_gaps, 1]  #xy
            Pyz_0 = P[i*ensemble_gaps, 2]  #yz
            Pzx_0 = P[i*ensemble_gaps, 3]  #zx
            F_0   = F[i*ensemble_gaps, 3]  #z coordinate

            for j in range(ensemble_gaps):
                ensemble_avg_P += Pxy_0*P[i*ensemble_gaps + j, 1] + Pyz_0*P[i*ensemble_gaps + j, 2] + Pzx_0*P[i*ensemble_gaps + j, 3]
                ensemble_avg_F += F_0*F[i*ensemble_gaps + j, 3]


        #average over ensembles
        ensemble_avg_P /= n_ensembles
        ensemble_avg_F /= n_ensembles

        #insert pint units
        ensemble_avg_P *= (ureg.kg * (ureg.angstrom / ureg.fs)**2)**2
        ensemble_avg_F *= (ureg.kg * ureg.angstrom / (ureg.fs**2))**2

        eta     = ensemble_avg_P*dt_*V3_kB_T_inv
        lambda_ = ensemble_avg_F*dt_*A_kB_T_inv

        return (eta.to(ureg.Pa*ureg.s)).magnitude, (lambda_.to(ureg.N * ureg.s / (ureg.m**3))).magnitude
