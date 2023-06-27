#!/usr/bin/env python
import mirheo as mir
import pint

if __name__ == "__main__":
    ureg = pint.UnitRegistry()

    # Define Mirheo's unit system.
    ureg.define('mirL = 1 angstrom')
    ureg.define('mirT = 1 fs')
    ureg.define('mirM = 1 kg')
    mir.set_unit_registry(ureg)


    dt = ureg('10 fs')            		   # timestep
    rc = ureg('0.432 nm')                  # cutoff radius
    number_density = ureg('33.32819504701638 nm^-3')   # 31.25 paper & 33.32772149 real Water density #33.327677048150235

    ranks  = (1, 1, 1)
    domain = (ureg('10.0 nm'), ureg('10.0 nm'), ureg('10.0 nm'))

    u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log')
    
    pv = mir.ParticleVectors.ParticleVector('pv', mass = ureg('18.015/6.02214076e23 g'))
    ic = mir.InitialConditions.Uniform(number_density)
    u.registerParticleVector(pv, ic)

    # Create and register sw interaction with specific parameters and cutoff radius
    sw2 = mir.Interactions.Pairwise('sw', rc, kind="SW", epsilon=ureg('6.189/6.02214076e23 kcal'), sigma=ureg('0.23925 nm'), A=7.049556277, B=0.6022245584)
    u.registerInteraction(sw2)
    u.setInteraction(sw2, pv, pv)

    sw3 = mir.Interactions.Triplewise('sw3', rc, kind='SW3', lambda_=23.15, epsilon=ureg('6.189/6.02214076e23 kcal'), theta=1.9106332362490186, gamma=1.2, sigma=ureg('0.23925 nm'))
    u.registerInteraction(sw3)
    u.setInteraction(sw3, pv, pv, pv)

    ##############################
    # Stage 1. stabilize with Minimize
    ##############################

    minimize = mir.Integrators.Minimize('int_minimize', max_displacement=ureg('0.1 nm'))
    u.registerIntegrator(minimize)
    u.setIntegrator(minimize, pv)

    u.registerPlugins(mir.Plugins.createStats('stats', every=10000))
    u.run(11, dt=dt)

    u.deregisterIntegrator(minimize)
    del minimize

    ##############################
    # Stage 2. vv with very small timesteps
    ##############################
    
    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)
    
    #Berendsen Thermostat							
    u.registerPlugins(mir.Plugins.createBerendsenThermostat('thermostat', [pv], T=ureg('300 K'), tau=ureg('1.0 ps')))


    u.run(21, dt=0.1*dt)
    
    ##############################
    # Stage 3. rdf run (like paper)
    ##############################
    path = "h5_water_test_truetrue/"
    dump_every = 1

    ####Viscosity####
    mask  = "111111111"
    u.registerPlugins(mir.Plugins.createStressTensor('stress_tensor', pv, dump_every, mask, path))

    ####Friction coef.####
    u.registerPlugins(mir.Plugins.createTotalForceSaver('tangential_force', pv, dump_every, path))


    u.run(101, dt=dt)
