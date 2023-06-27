#!/usr/bin/env python
import mirheo as mir
import pint
import numpy as np

#simulating Argon to check if viscosity is correctly calculated (Kirova et al. 2015)

if __name__ == "__main__":
    ureg = pint.UnitRegistry()

    # Define Mirheo's unit system.
    ureg.define('mirL = 1 angstrom')
    ureg.define('mirT = 1 fs')
    ureg.define('mirM = 1 kg')
    mir.set_unit_registry(ureg)


    dt = ureg('5 fs')           		   # timestep (normally 0.05 ps)
    rc = ureg('1.0 nm')                    # cutoff radius (this is chosen on my side)

    number_density = 1.3507*(6.02214076e23/39.948)*ureg('cm^-3')    #Density of argon at 94.4K https://lar.bnl.gov/properties/

    ranks  = (1, 1, 1)
    domain = (ureg('10.0 nm'), ureg('10.0 nm'), ureg('10.0 nm'))

    u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log')
    
    pv = mir.ParticleVectors.ParticleVector('pv', mass = ureg('39.948/6.02214076e23 g'))
    ic = mir.InitialConditions.Uniform(number_density)
    u.registerParticleVector(pv, ic)

    # Create and register lj interaction with specific parameters and cutoff radius
    lj = mir.Interactions.Pairwise('lj', rc, kind='LJ', epsilon=ureg('120*1.38064852e-23 J'), sigma=ureg('3.4 angstrom'))
    u.registerInteraction(lj)
    u.setInteraction(lj, pv, pv)

    ##############################
    # Stage 1. stabilize with Minimize
    ##############################

    minimize = mir.Integrators.Minimize('int_minimize', max_displacement=ureg('0.1 nm'))
    u.registerIntegrator(minimize)
    u.setIntegrator(minimize, pv)

    u.registerPlugins(mir.Plugins.createStats('stats', every=10000))

    path = "h5_argon/"
    #u.registerPlugins(mir.Plugins.createDumpParticles('part_dump', pv, 10000, [], path + "pv-"))
    
    u.run(10001, dt=dt)

    u.deregisterIntegrator(minimize)
    del minimize

    ##############################
    # Stage 2. vv with very small timesteps
    ##############################
    
    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)
    
    #Berendsen Thermostat							
    u.registerPlugins(mir.Plugins.createBerendsenThermostat('thermostat', [pv], T=ureg('94.4 K'), tau=ureg('1.0 ps')))


    u.run(200001, dt=dt)
    
    ##############################
    # Stage 3. rdf run (like paper)
    ##############################
    
    dump_every = 1

    ####Viscosity####
    mask  = "011101110" #I don't want off-diagonals
    u.registerPlugins(mir.Plugins.createStressTensor('stress_tensor', pv, dump_every, mask, path))

    ####Friction coef.####
    u.registerPlugins(mir.Plugins.createTotalForceSaver('tangential_force', pv, dump_every, path))


    u.run(1000001, dt=dt)
