import mirheo as mir
import pint
import numpy as np
import os
import sys
import shutil
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#simulating water droplet on graphene sheet to measure the contact angle

if __name__ == "__main__":
    ureg = pint.UnitRegistry()

    # Define Mirheo's unit system.
    ureg.define('mirL = 1 angstrom')
    ureg.define('mirT = 1 fs')
    ureg.define('mirM = 1e-23 g')
    mir.set_unit_registry(ureg)


    #load graphene particle positions
    droplet = np.loadtxt("particle_generators/water_cube_16384.txt")
    grs = np.loadtxt("particle_generators/grs_2_200_200.txt")


    droplet_n = droplet.shape[0]
    shift = np.array([150.0, 150.0, 16.8])  #c-w ~= [2.8,3.3] angstrom
    for i in range(droplet_n):
        droplet[i,:] += shift

    grs_n = grs.shape[0]
    shift = np.array([100.0, 100.0, 10.1])
    for i in range(grs_n):
        grs[i,:] += shift

    
    dt = ureg('2 fs')            		   # timestep   (petros 2003)
    rc_ww = ureg('0.432 nm')
    rc_cw = ureg('10.0 angstrom')          #10 A was normaly used in petros 2003

    ranks  = (1, 1, 1)
    domain = (ureg('400.0 angstrom'), ureg('400.0 angstrom'), ureg('100.0 angstrom'))

    u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log')
    
    #register particles
    pv_w = mir.ParticleVectors.ParticleVector('pv_w', mass=ureg('18.015/6.02214076e23 g'))
    ic = mir.InitialConditions.FromArray( pos=droplet, vel=np.random.normal(scale=372e-5, size=(droplet_n,3)))
    u.registerParticleVector(pv_w, ic)

    pv_c = mir.ParticleVectors.ParticleVector('pv_g', mass=ureg('12.0107/6.02214076e23 g'))
    #pv_c = mir.ParticleVectors.ParticleVector('pv_g', mass=ureg('1e15 g'))    #make the mass inf such that carbon atoms are "frozen"
    ic = mir.InitialConditions.FromArray( pos=grs, vel=np.zeros((grs_n,3)))
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
    u.registerPlugins(mir.Plugins.createDumpParticles('part_dump_c', pv_c, 30*dump_every, [], 'h5_droplet_grs/pv_c-'))
    u.registerPlugins(mir.Plugins.createDumpParticles('part_dump_w', pv_w, dump_every, [], 'h5_droplet_grs/pv_w-'))

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
    u.run(30002, dt=dt)
