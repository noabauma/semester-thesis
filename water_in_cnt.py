import mirheo as mir
import pint
import numpy as np
import h5py
# import os
# import sys
# import shutil
#import glob
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#simulating water droplet on graphene sheet to measure the contact angle

if __name__ == "__main__":
    ureg = pint.UnitRegistry()

    # Define Mirheo's unit system.
    ureg.define('mirL = 1 angstrom')
    ureg.define('mirT = 1 fs')
    ureg.define('mirM = 1 kg')
    mir.set_unit_registry(ureg)


    #load particles
    h5file = True
    if h5file:
        water = h5py.File('particle_generators/water_for_cnt_96_0_200/pv.PV-00000.h5', 'r')
        w_pos = water["position"][()]
        w_vel = water["velocity"][()]

        c_pos = h5py.File('particle_generators/h5_cnt_96_0_200/pv_c-00001.h5', 'r')["position"][()]
        Ncnt = c_pos.shape[0]
        c_vel = np.zeros((Ncnt, 3), dtype=np.float64)
    else:
        particles = np.loadtxt("/home/noabauma/semesterthesis/particle_generators/cnt_water_96_0_200.xyz", dtype=str, skiprows=2)
        w_pos = particles[particles[:,0] == 'W',1:].astype(np.float64)
        Nwater = w_pos.shape[0]
        w_vel = np.random.normal(scale=372e-5, size=(Nwater,3))

        c_pos = particles[particles[:,0] == 'C',1:].astype(np.float64)
        Ncnt = c_pos.shape[0]
        c_vel = np.zeros((Ncnt, 3), dtype=np.float64)

    #determine the space for periodic boundary
    ac = False                                                                  #armchair or zigzag pattern in horizontal layer?
    L  = np.max(c_pos[:,2]) - np.min(c_pos[:,2])                                #length of the cnt [angstrom]
    L = (L + 1.23353) * ureg.angstrom if ac else (L + 1.42436) * ureg.angstrom  # 1.23353 = 1.42436*cos(30°)


    #shift particles
    shift = np.array([200.0 - np.mean(c_pos[:,0]), 200.0 - np.mean(c_pos[:,1]), 0.1 - np.min(c_pos[:,2])])
    c_pos += shift
    shift = np.array([200.0 - np.mean(w_pos[:,0]), 200.0 - np.mean(w_pos[:,1]), 0.1 - np.min(w_pos[:,2])])
    w_pos += shift

    
    dt = ureg('2 fs')            		   # timestep   (petros 2003)
    rc_ww = ureg('0.432 nm')
    rc_cw = ureg('10.0 angstrom')          #10 A was normaly used in petros 2003

    ranks  = (1, 1, 1)
    domain = (ureg('400.0 angstrom'), ureg('400.0 angstrom'), L)
    #domain = (ureg('400.0 angstrom'), ureg('400.0 angstrom'), ureg('220.0 angstrom'))

    u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log')
    
    #register particles
    pv_w = mir.ParticleVectors.ParticleVector('pv_w', mass=ureg('18.015/6.02214076e23 g'))
    ic = mir.InitialConditions.FromArray( pos=w_pos, vel=w_vel)
    u.registerParticleVector(pv_w, ic)

    pv_w_cp = mir.ParticleVectors.ParticleVector('pv_w_cp', mass=ureg('18.015/6.02214076e23 g'))  
    #ic = mir.InitialConditions.FromArray( pos=w_pos, vel=w_vel)
    u.registerParticleVector(pv_w_cp, ic) #to save w-c interaction separat to subtract F_i from pv_w

    pv_c = mir.ParticleVectors.ParticleVector('pv_c', mass=ureg('12.0107/6.02214076e23 g'))
    ic = mir.InitialConditions.FromArray( pos=c_pos, vel=c_vel)
    u.registerParticleVector(pv_c, ic)


    ##############################
    # Interactions
    ##############################  

    ###water-water###
    sw3_ww = mir.Interactions.Triplewise('sw3_ww', rc_ww, kind='SW3', lambda_=23.15, epsilon=ureg('6.189/6.02214076e23 kcal'), theta=1.9106332362490186, gamma=1.2, sigma=ureg('0.23925 nm'))
    sw2_ww = mir.Interactions.Pairwise('sw2_ww', rc_ww, kind='SW', epsilon=ureg('6.189/6.02214076e23 kcal'), sigma=ureg('0.23925 nm'), A=7.049556277, B=0.6022245584)
    u.registerInteraction(sw3_ww)
    u.registerInteraction(sw2_ww)
    u.setInteraction(sw3_ww, pv_w, pv_w, pv_w)
    u.setInteraction(sw2_ww, pv_w, pv_w)
    

    ###carbon-water###
    lj_cw = mir.Interactions.Pairwise('lj_cw', rc_cw, kind='LJ', epsilon=ureg('0.392/6.02214076e23 kJ'), sigma=ureg('3.19 angstrom'))   #Petros 2003
    u.registerInteraction(lj_cw)
    u.setInteraction(lj_cw, pv_c, pv_w)

    u.setInteraction(lj_cw, pv_c, pv_w_cp)  #force saver for subtraction

    """
    sw2_cw = mir.Interactions.Pairwise('sw2_cw', rc_ww, kind='SW', stress=True, stress_period=1, epsilon=ureg('0.3651/6.02214076e23 kJ'), sigma=ureg('3.19 angstrom'), A=7.049556277, B=0.6022245584)    #Jaffe et al. 2004
    u.registerInteraction(sw2_cw)
    u.setInteraction(sw2_cw, pv_c, pv_w)
    """
    


    ##############################
    # Stage 1. stabilize with Minimize
    ##############################
    minimize = mir.Integrators.Minimize('int_minimize', max_displacement=ureg('1.0 angstrom'))
    u.registerIntegrator(minimize)
    u.setIntegrator(minimize, pv_w)

    u.registerPlugins(mir.Plugins.createStats('stats', every=10000))

    u.run(11, dt=dt)

    u.deregisterIntegrator(minimize)
    del minimize

    ##############################
    # Stage 2. Velocity-Verlet & get temperature right (equilibrium) (should take 0.2ns)
    ##############################
    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv_w)
    

    ####Plugins####
    u.registerPlugins(mir.Plugins.createCopyPV('copy_pv', pv_w_cp, pv_w))				
    u.registerPlugins(mir.Plugins.createBerendsenThermostat('thermostat', [pv_w], T=ureg('300.0 K'), tau=ureg('2.0 ps')))   #petros 2003
    
    u.run(11, dt=dt)

    ##############################
    # Stage 3. Save stress tensor & total tangential force with Miheo plugins into csv files
    ##############################
    #debug
    path = "h5_w_in_cnt1111/"
    dump_every = 1
    u.registerPlugins(mir.Plugins.createDumpParticles('part_dump_w', pv_w, dump_every, [], path + "pv_w-"))
    u.registerPlugins(mir.Plugins.createDumpParticles('part_dump_c', pv_c, dump_every, [], path + "pv_c-"))

    
    dump_every = 1
    ####Viscosity#### 
    mask = '011101110'   #"Pxx", "Pxy", "Pxz", "Pyx", "Pyy", "Pyz", "Pzx", "Pzy", "Pzz"
    u.registerPlugins(mir.Plugins.createStressTensor('stress_tensor', pv_w, dump_every, mask, path))
    u.registerPlugins(mir.Plugins.createStressTensor('stress_tensor_cp', pv_w_cp, dump_every, mask, path))

    ####Friction coef.####
    u.registerPlugins(mir.Plugins.createTotalForceSaver('tangential_force', pv_w, dump_every, path))
    

    u.run(11, dt=dt)
