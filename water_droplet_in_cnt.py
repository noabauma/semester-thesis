import mirheo as mir
import pint
import numpy as np
import os
import sys
import shutil
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#simulating water droplet in CNT to measure the contact angle

if __name__ == "__main__":
    ureg = pint.UnitRegistry()

    # Define Mirheo's unit system.
    ureg.define('mirL = 1 angstrom')
    ureg.define('mirT = 1 fs')
    ureg.define('mirM = 1e-23 g')
    mir.set_unit_registry(ureg)


    #load graphene particle positions
    #droplet = np.loadtxt("particle_generators/water_droplet_69.txt")
    droplet = np.loadtxt("particle_generators/water_cylinder_12600.txt")
    cnt = np.loadtxt("particle_generators/cnt_96_0_200.txt")

    R_droplet = (np.max(droplet[:,0]) - np.min(droplet[:,0]) + np.max(droplet[:,1]) - np.min(droplet[:,1]) + np.max(droplet[:,2]) - np.min(droplet[:,2]))/6.0
    R_cnt = (np.max(cnt[:,0]) - np.min(cnt[:,0]) + np.max(cnt[:,1]) - np.min(cnt[:,1]))*0.25
    L_cnt = np.max(cnt[:,2]) - np.min(cnt[:,2])

    #shifting droplet & cnt to the desired position
    droplet_n = droplet.shape[0]
    shift = np.array([200.0, 200.0, 200.0])     #Radius of water droplet is not bigger than Radius of CNT! [R_droplet < R_cnt - (c-w)]
    for i in range(droplet_n):
        droplet[i,:] += shift

    cnt_n = cnt.shape[0]
    shift = np.array([200.0, 200.0, 200.0 - L_cnt*0.5])
    for i in range(cnt_n):
        cnt[i,:] += shift

    
    dt = ureg('2 fs')            		   # timestep   (petros 2003)
    rc_ww = ureg('0.432 nm')
    rc_cw = ureg('10.0 angstrom')          #10 A was normaly used in petros 2003

    ranks  = (1, 1, 1)
    domain = (ureg('400.0 angstrom'), ureg('400.0 angstrom'), ureg('400.0 angstrom'))

    u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log')
    
    #register particles
    pv_w = mir.ParticleVectors.ParticleVector('pv_w', mass=ureg('18.015/6.02214076e23 g'))
    ic = mir.InitialConditions.FromArray( pos=droplet, vel=np.random.normal(scale=372e-5, size=(droplet_n,3)))
    u.registerParticleVector(pv_w, ic)

    pv_c = mir.ParticleVectors.ParticleVector('pv_g', mass=ureg('12.0107/6.02214076e23 g'))
    ic = mir.InitialConditions.FromArray( pos=cnt, vel=np.zeros((cnt_n,3)))
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

    """
    sw2_cw = mir.Interactions.Pairwise('sw2_cw', rc_ww, kind='SW', epsilon=ureg('0.3651/6.02214076e23 kJ'), sigma=ureg('3.19 angstrom'), A=7.049556277, B=0.6022245584)    #Jaffe et al. 2004
    u.registerInteraction(sw2_cw)
    u.setInteraction(sw2_cw, pv_c, pv_w)
    """
    


    ##############################
    # Stage 1. stabilize with Minimize
    ##############################
    minimize = mir.Integrators.Minimize('int_minimize', max_displacement=ureg('1.0 angstrom'))
    u.registerIntegrator(minimize)
    u.setIntegrator(minimize, pv_w)

    plugin_stats = mir.Plugins.createStats('stats', every=2000)
    u.registerPlugins(plugin_stats)

    #debug
    dump_every = 1000
    u.registerPlugins(mir.Plugins.createDumpParticles('part_dump_c', pv_c, 30*dump_every, [], 'h5_droplet_cnt/pv_c-'))
    u.registerPlugins(mir.Plugins.createDumpParticles('part_dump_w', pv_w, dump_every, [], 'h5_droplet_cnt/pv_w-'))

    u.run(20002, dt=dt)

    u.deregisterPlugins(plugin_stats)
    u.deregisterIntegrator(minimize)
    del plugin_stats
    del minimize

    ##############################
    # Stage 2. Velocity-Verlet
    ##############################
    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv_w)
    

    ####Plugins####				
    u.registerPlugins(mir.Plugins.createBerendsenThermostat('thermostat', [pv_w], T=ureg('300.0 K'), tau=ureg('2.0 ps')))   #petros 2003
    u.registerPlugins(mir.Plugins.createStats('stats', every=1000))
    
    

    
    ###############
    u.run(100002, dt=dt)
