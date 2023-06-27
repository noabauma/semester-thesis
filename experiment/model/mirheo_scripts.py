from json import dump
import mirheo as mir
import pint
import numpy as np
from scipy.optimize import curve_fit
import h5py
import glob
from mpi4py import MPI

debug = True

###
#These are all of the mirheo script for korali
###

#sigmoidal curve for fitting the density function
def sigmoidal(z, rho_l, z_e, d):
    #rho_l = 0.033328        #density of water [0.0167, 0.0368]
    return rho_l*0.5*(1.0 - np.tanh(2.0*(z - z_e)/d))


########################
# Preload everything for all Mirheo scripts
########################
ureg = pint.UnitRegistry()

# Define Mirheo's unit system.
ureg.define('mirL = 1 angstrom')
ureg.define('mirT = 1 fs')
ureg.define('mirM = 1 kg')
mir.set_unit_registry(ureg)

dt = ureg('2 fs')            		    #timestep   (Werder et al. 2003)
rc_ww = ureg('0.432 nm')                #(Molinero et al. 2009)

ranks  = (1, 1, 1)
domain = (ureg('400.0 angstrom'), ureg('400.0 angstrom'), ureg('400.0 angstrom'))


#load particle positions for WCA GRS
grs_droplet = np.loadtxt("particles/water_cube_16384.txt")
grs = np.loadtxt("particles/grs_2_200_200.txt")

grs_droplet_n = grs_droplet.shape[0]
shift = np.array([150.0, 150.0, 16.8])  #c-w ~= [2.6,3.3] angstrom
grs_droplet += shift
grs_droplet_vel = np.random.normal(scale=372e-5, size=(grs_droplet_n,3))

grs_n = grs.shape[0]
shift = np.array([100.0, 100.0, 10.1])
grs += shift
grs_vel = np.zeros((grs_n,3))


#load particle positions for WCA CNT
#droplet = np.loadtxt("particle_generators/water_droplet_69.txt")
cnt_droplet = np.loadtxt("particles/water_cylinder_12600.txt")
cnt = np.loadtxt("particles/cnt_96_0_200.txt")

L_cnt = np.max(cnt[:,2]) - np.min(cnt[:,2])

#shifting droplet & cnt to the desired position
cnt_droplet_n = cnt_droplet.shape[0]
shift = np.array([200.0, 200.0, 200.0])     #Radius of water droplet is not bigger than Radius of CNT! [R_droplet < R_cnt - (c-w)]
cnt_droplet += shift
cnt_droplet_vel = np.random.normal(scale=372e-5, size=(cnt_droplet_n,3))    #scale is in angstrom/fs

cnt_n = cnt.shape[0]
shift = np.array([200.0, 200.0, 200.0 - L_cnt*0.5])
cnt += shift
cnt_vel = np.zeros((cnt_n,3))



#Friction ceof & viscosity parameters
#load particles for Water in CNT
water = h5py.File('particles/water_for_cnt_96_0_200/pv.PV-00000.h5', 'r')
w_pos = water["position"][()]
w_vel = water["velocity"][()]

c_pos = h5py.File('particles/h5_cnt_96_0_200/pv_c-00001.h5', 'r')["position"][()]
Ncnt = c_pos.shape[0]
c_vel = np.zeros((Ncnt, 3), dtype=np.float64)

shift = np.array([100.0 - np.mean(c_pos[:,0]), 100.0 - np.mean(c_pos[:,1]), 0.1 - np.min(c_pos[:,2])])
c_pos += shift
shift = np.array([100.0 - np.mean(w_pos[:,0]), 100.0 - np.mean(w_pos[:,1]), 0.1 - np.min(w_pos[:,2])])
w_pos += shift

#determine the space for periodic boundary
ac = False                                                                  #armchair or zigzag pattern in horizontal layer?
L_domain = (L_cnt + 1.23353) * ureg.angstrom if ac else (L_cnt + 1.42436) * ureg.angstrom  # 1.23353 = 1.42436*cos(30°)

domain_frct_visc = (ureg('200.0 angstrom'), ureg('200.0 angstrom'), L_domain)



#parameters
kB = 1.38064852e-23 * ureg.kg * (ureg.m/ureg.s)**2 / ureg.K     #Boltzmann constant            [kg * (m/s)^2 * K^-1]
T  = 300.0 * ureg.K                                             #Temperature in the simulation [K]

R = 75.392*0.5 * ureg.angstrom  #Radius of the CNT
L = L_cnt      * ureg.angstrom  #Length of the CNT
V = np.pi*R*R*L                 #Volume of the domain or CNT (or non if formula (13) is taken)?
A = 2.0*np.pi*R*L               #Surface inside the CNT

V3_kB_T_inv = 1.0/(3.0*V*kB*T)
A_kB_T_inv = 1.0/(A*kB*T)



########################
# WCA on GRS
########################
def droplet_grs(A_eps, B, sigma, rc_cw):
    ##############################
    # Stage 0. Initialize Mirheo
    ##############################
    u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', comm_ptr=MPI._addressof(MPI.COMM_WORLD))

    #register particles
    pv_w = mir.ParticleVectors.ParticleVector('pv_w', mass=ureg('18.015/6.02214076e23 g'))
    ic = mir.InitialConditions.FromArray( pos=grs_droplet, vel=grs_droplet_vel)
    u.registerParticleVector(pv_w, ic)

    pv_c = mir.ParticleVectors.ParticleVector('pv_c', mass=ureg('12.0107/6.02214076e23 g'))
    ic = mir.InitialConditions.FromArray( pos=grs, vel=grs_vel)
    u.registerParticleVector(pv_c, ic)

    ###water-water interaction (Molinero et al. 2009)###
    sw3_ww = mir.Interactions.Triplewise('sw3_ww', rc_ww, kind='SW3', lambda_=23.15, epsilon=ureg('6.189/6.02214076e23 kcal'), theta=1.9106332362490186, gamma=1.2, sigma=ureg('0.23925 nm'))
    sw2_ww = mir.Interactions.Pairwise('sw2_ww', rc_ww, kind='SW', epsilon=ureg('6.189/6.02214076e23 kcal'), sigma=ureg('0.23925 nm'), A=7.049556277, B=0.6022245584)
    u.registerInteraction(sw3_ww)
    u.registerInteraction(sw2_ww)
    u.setInteraction(sw3_ww, pv_w, pv_w, pv_w)
    u.setInteraction(sw2_ww, pv_w, pv_w)

    ###carbon-water interaction###
    sw2_cw = mir.Interactions.Pairwise('sw2_cw', rc_cw, kind='SW', epsilon=A_eps*ureg.J, sigma=sigma*ureg.angstrom, A=1.0, B=B)    #Koralis values
    u.registerInteraction(sw2_cw)
    u.setInteraction(sw2_cw, pv_c, pv_w)


    ##############################
    # Stage 1. stabilize with Minimize
    ##############################
    minimize = mir.Integrators.Minimize('int_minimize', max_displacement=ureg('1.0 angstrom'))
    u.registerIntegrator(minimize)
    u.setIntegrator(minimize, pv_w)

    #debug
    if debug:
        u.registerPlugins(mir.Plugins.createStats('stats', every=200))
        u.run(2000, dt=dt)
    else:
        u.run(20000, dt=dt)

    u.deregisterIntegrator(minimize)
    del minimize

    ##############################
    # Stage 2. Velocity-Verlet
    ##############################
    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv_w)
    
    ####Plugins####				
    u.registerPlugins(mir.Plugins.createBerendsenThermostat('thermostat', [pv_w], T=T, tau=ureg('2.0 ps')))   #(Werder et al. 2003)
   
    if debug:
        u.run(2002, dt=dt)
    else:
        u.run(100000, dt=dt)

    ##############################
    # Stage 3. Dump-Particles
    ##############################
    if debug:
        dump_every = 100
        u.registerPlugins(mir.Plugins.createDumpParticles('part_dump_w', pv_w, dump_every, [], 'data/h5_droplet_grs/pv_w-'))

        u.run(1002, dt=dt)
    else:
        dump_every = 1000
        u.registerPlugins(mir.Plugins.createDumpParticles('part_dump_w', pv_w, dump_every, [], 'data/h5_droplet_grs/pv_w-'))

        u.run(10002, dt=dt)

    ##############################
    # Stage 4. Calculate the WCA
    ##############################
    if u.isComputeTask():
        wca_tot = 0.0
        
        ###STEP 4.0: load particles in either debug mode or from h5file###
        all_files = glob.glob("data/h5_droplet_grs/pv_w-*.h5")
        all_files = np.sort(all_files)
        n_files   = len(all_files)
        for file_ in range(n_files):
            particles = h5py.File(all_files[file_], "r")["position"][()]
            
            particles_x = particles[:,0]
            particles_y = particles[:,1]
            particles_z = particles[:,2]

            d = (np.max(particles_x) - np.min(particles_x) + np.max(particles_y) - np.min(particles_y) ) * 0.5
            r = d*0.5
            
            mean_x = np.mean(particles_x)
            mean_y = np.mean(particles_y)
            min_z  = np.min(particles_z)
        
            ###shift droplet to the center###
            shift = np.array([mean_x, mean_y, min_z])
            particle = np.empty((grs_droplet_n,2), dtype=np.float64)             #particle container which holds [:,0]=xy-l2-distance & [:,1]=z-coordinate 
            for i in range(grs_droplet_n):
                particles[i,:] -= shift
                particle[i,0]  = particles[i,0]*particles[i,0] + particles[i,1]*particles[i,1]
                particle[i,1]  = particles[i,2]
            
            c_w = min(abs(min_z - 13.5), 5.0)   #bond distance between water and graphene sheet

            ###bins properties###
            h_bins = 10.0                                            #height of bins in angstrom [5.0 - 10.0 does also give promising results (but is 2x slower)]
            dA = 95.0                                                #base area per bin [A^2]
            r_bins = np.sqrt(dA/np.pi)                               #bin radius
            r_bins_half = r_bins*0.5
            h_pi_4_r_bins = h_bins*np.pi*4.0*r_bins_half             #base volume of a bin
            
            M_bins = 400                                             #number of bins in z-direction  [the more the better but also slower]
            bins_step = np.linspace(0.0, np.max(particle[:,1])*1.1, M_bins)
            
            z_0 = 8.0                                                #Cut-off [angstrom] any points smaller than z_0 will be neglected
            z_0_loc = 0                                              #location where z_0 begins
            for j in range(M_bins):
                if bins_step[j] >= z_0:
                    z_0_loc = j
                    break
        
            N_bins = 100                                            #number of bins in xy-direction [the more the better but also slower]
            bins_loc = np.linspace(r_bins_half, r*0.75, N_bins)     #0.50-0.80 are good values

            r2 = np.empty((N_bins,4), dtype=np.float64)
            for i in range(N_bins):
                r = bins_loc[i]
                r2[i,0] = (r - r_bins_half)*(r - r_bins_half)
                r2[i,1] = (r + r_bins_half)*(r + r_bins_half)
                r2[i,2] = 1.0/(h_pi_4_r_bins*r)



            ###STEP 4.1: calculate all the M_bins (xy-direction) density profiles###
            density_profile = np.empty((N_bins,M_bins), dtype=np.float64)
            for j in range(M_bins):
                bin_step = bins_step[j]
                
                particles_in_h_layer = particle[particle[:,1] >= bin_step, :]
                particles_in_h_layer = particles_in_h_layer[particles_in_h_layer[:,1] < (bin_step + h_bins), :]

                for i in range(N_bins):
                    r2_low  = r2[i,0]
                    r2_high = r2[i,1]
                    inv_V   = r2[i,2]
                    
                    p_r2 = particles_in_h_layer[:,0]
                    count = np.count_nonzero((r2_low <= p_r2) & (p_r2 <= r2_high))
                    
                    density_profile[i,j] = float(count) * inv_V  #[angstrom^-3]
            


            ###STEP 4.2: sigmoidal curve fit###
            sigmoidal_fit = []                                     #container of all sigmoidal z_e points
            for i in range(N_bins):
                densities = density_profile[i,z_0_loc:]
                bins = bins_step[z_0_loc:]    
                
                keep = (0.0167 <= densities) & (densities <= 0.0368)
                densities = densities[keep]
                bins      = bins[keep]

                if densities.shape[0] > 0:
                    popt, pcov = curve_fit(sigmoidal, bins, densities, p0=[0.033328, r, 4.3], bounds=([0.0167, 0.0, 0.1], [0.0368, d, 50.0]))
                    sigmoidal_fit.append([bins_loc[i], popt[1]])



            ###STEP 4.3: Fit circular curve (find z_f and r_f)###
            sigmoidal_fit = np.array(sigmoidal_fit)
            n_sig = sigmoidal_fit.shape[0]

            if n_sig == 0:                          #continue if non fitted this also means wca = 0°
                continue

            temp = np.vstack([sigmoidal_fit[:,0], np.ones(n_sig)]).T

            z_f = np.max(particle[:,1])
            while(True):
                trns = sigmoidal_fit[:,0]**2 + (sigmoidal_fit[:,1] - z_f)**2
            
                slope, radius = np.linalg.lstsq(temp, trns, rcond=None)[0]
                
                if slope < 1e-5  or z_f < -4000.0:
                    break

                z_f -= min(slope, r)


            ###STEP 4.4: calculate the WCA###
            z_f += c_w                         #added spacing between w-c
            r_f = np.sqrt(radius)
                
            wca = np.arcsin(z_f/r_f) + np.pi*0.5 if z_f > 0.0 else np.pi*0.5 - np.arcsin(abs(z_f)/r_f)

            wca_tot += wca

            if debug:
                print("z_f = ", z_f, ", slope = ", slope, ", r_f = ", r_f)
                print("wca = ", wca*(180.0/np.pi), " [", file_, "]", flush=True)

        return wca_tot*(180.0/np.pi)/n_files







########################
# WCA in CNT
########################
def droplet_cnt(A_eps, B, sigma, rc_cw):
    ##############################
    # Stage 0. Initialize Mirheo
    ##############################
    u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', comm_ptr=MPI._addressof(MPI.COMM_WORLD))

    #register particles
    pv_w = mir.ParticleVectors.ParticleVector('pv_w', mass=ureg('18.015/6.02214076e23 g'))
    ic = mir.InitialConditions.FromArray( pos=cnt_droplet, vel=cnt_droplet_vel)
    u.registerParticleVector(pv_w, ic)

    pv_c = mir.ParticleVectors.ParticleVector('pv_c', mass=ureg('12.0107/6.02214076e23 g'))
    ic = mir.InitialConditions.FromArray( pos=cnt, vel=cnt_vel)
    u.registerParticleVector(pv_c, ic)

    ###water-water interaction (Molinero et al. 2009)###
    sw3_ww = mir.Interactions.Triplewise('sw3_ww', rc_ww, kind='SW3', lambda_=23.15, epsilon=ureg('6.189/6.02214076e23 kcal'), theta=1.9106332362490186, gamma=1.2, sigma=ureg('0.23925 nm'))
    sw2_ww = mir.Interactions.Pairwise('sw2_ww', rc_ww, kind='SW', epsilon=ureg('6.189/6.02214076e23 kcal'), sigma=ureg('0.23925 nm'), A=7.049556277, B=0.6022245584)
    u.registerInteraction(sw3_ww)
    u.registerInteraction(sw2_ww)
    u.setInteraction(sw3_ww, pv_w, pv_w, pv_w)
    u.setInteraction(sw2_ww, pv_w, pv_w)

    ###carbon-water interaction###
    sw2_cw = mir.Interactions.Pairwise('sw2_cw', rc_cw, kind='SW', epsilon=A_eps*ureg.J, sigma=sigma*ureg.angstrom, A=1.0, B=B)    #Koralis values
    u.registerInteraction(sw2_cw)
    u.setInteraction(sw2_cw, pv_c, pv_w)


    ##############################
    # Stage 1. stabilize with Minimize
    ##############################
    minimize = mir.Integrators.Minimize('int_minimize', max_displacement=ureg('1.0 angstrom'))
    u.registerIntegrator(minimize)
    u.setIntegrator(minimize, pv_w)

    if debug:
        u.registerPlugins(mir.Plugins.createStats('stats', every=200))

        u.run(2002, dt=dt)
    else:
        u.run(20000, dt=dt)

    u.deregisterIntegrator(minimize)
    del minimize

    ##############################
    # Stage 2. Velocity-Verlet
    ##############################
    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv_w)
    
    ####Plugins####				
    u.registerPlugins(mir.Plugins.createBerendsenThermostat('thermostat', [pv_w], T=T, tau=ureg('2.0 ps')))   #(Werder et al. 2003)
    
    if debug:
        u.run(2002, dt=dt)
    else:
        u.run(100000, dt=dt)

    ##############################
    # Stage 3. Dump-Particles
    ##############################
    if debug:
        dump_every = 100
        u.registerPlugins(mir.Plugins.createDumpParticles('part_dump_w', pv_w, dump_every, [], 'data/h5_droplet_cnt/pv_w-'))

        u.run(1002, dt=dt)
    else:
        dump_every = 1000
        u.registerPlugins(mir.Plugins.createDumpParticles('part_dump_w', pv_w, dump_every, [], 'data/h5_droplet_cnt/pv_w-'))

        u.run(10002, dt=dt)


    ##############################
    # Stage 4. Calculate the WCA
    ##############################
    if u.isComputeTask():
        wca_tot = 0.0

        ###STEP 4.0: load particles in either debug mode or from h5file###
        all_files = glob.glob("data/h5_droplet_cnt/pv_w-*.h5")
        all_files = np.sort(all_files)
        n_files   = len(all_files)
        for file_ in range(n_files):
            particles = h5py.File(all_files[file_], "r")["position"][()]
            
            particles_x = particles[:,0]
            particles_y = particles[:,1]
            particles_z = particles[:,2]

            d = (np.max(particles_x) - np.min(particles_x) + np.max(particles_y) - np.min(particles_y) ) * 0.5
            d_ = d*0.5                                      #Radius from Werder et al. 2001
            
            mean_x = np.mean(particles_x)
            mean_y = np.mean(particles_y)
            mean_z = np.mean(particles_z)

            #shift droplet to the center
            particle = np.empty((cnt_droplet_n,2), dtype=np.float64)
            shift = np.array([mean_x, mean_y, mean_z])
            for i in range(cnt_droplet_n):
                particles[i,:] -= shift
                particle[i,0] = particles[i,0]*particles[i,0] + particles[i,1]*particles[i,1]
                particle[i,1] = abs(particles[i,2])  
            

            #bins properties
            h_bins = 10.0                #height of bins in angstrom
            dA = 95.0                    #base area per bin angstrom^2
            r_bins = np.sqrt(dA/np.pi)   #bin radius
            r_bins_half = r_bins*0.5

            h_pi_8_r_bins = h_bins*np.pi*8.0*r_bins_half

            M_bins = 400                                             #number of bins in z-direction  [the more the better but also slower]
            bins_step = np.linspace(0.0, np.max(particle[:,1])*1.1, M_bins)
            
            z_0 = 0.0                                                #Cut-off [angstrom] any points smaller than z_0 will be neglected
            z_0_loc = 0                                              #location where z_0 begins
            for j in range(M_bins):
                if bins_step[j] >= z_0:
                    z_0_loc = j
                    break
        
            N_bins = 100                                            #number of bins in xy-direction [the more the better but also slower]
            bins_loc = np.linspace(r_bins_half, d_*0.8, N_bins)     #0.6-0.8 are good values

            r2 = np.empty((N_bins,4), dtype=np.float64)
            for i in range(N_bins):
                r = bins_loc[i]
                r2[i,0] = (r - r_bins_half)*(r - r_bins_half)
                r2[i,1] = (r + r_bins_half)*(r + r_bins_half)
                r2[i,2] = 1.0/(h_pi_8_r_bins*r)



            ###STEP 4.1: calculate all the M_bins (xy-direction) density profiles###
            density_profile = np.empty((N_bins,M_bins), dtype=np.float64)
            for j in range(M_bins):
                bin_step = bins_step[j]
                
                particles_in_h_layer = particle[particle[:,1] >= bin_step, :]
                particles_in_h_layer = particles_in_h_layer[particles_in_h_layer[:,1] < (bin_step + h_bins), :]

                for i in range(N_bins):
                    r2_low  = r2[i,0]
                    r2_high = r2[i,1]
                    inv_V   = r2[i,2]
                    
                    p_r2 = particles_in_h_layer[:, 0]
                    count = np.count_nonzero((r2_low <= p_r2) & (p_r2 <= r2_high))
                    
                    density_profile[i,j] = float(count) * inv_V  #[angstrom^-3]
            


            ###STEP 4.2: sigmoidal curve fit###
            sigmoidal_fit = []                                     #container of all sigmoidal z_e points
            for i in range(N_bins):
                densities = density_profile[i,z_0_loc:]
                bins = bins_step[z_0_loc:]    
                
                keep = (0.0167 <= densities) & (densities <= 0.0368)
                densities = densities[keep]
                bins      = bins[keep]

                if densities.shape[0] > 0:
                    popt, pcov = curve_fit(sigmoidal, bins, densities, p0=[0.033328, r, 4.3], bounds=([0.0167, 0.0, 0.1], [0.0368, d, 50.0]))
                    sigmoidal_fit.append([bins_loc[i], popt[1]])



            ###STEP 4.3: Fit circular curve (find z_f and r_f)###
            sigmoidal_fit = np.array(sigmoidal_fit)
            n_sig = sigmoidal_fit.shape[0]

            if n_sig == 0:  #continue if non fitted this also means wca = 0
                continue

            temp = np.vstack([sigmoidal_fit[:,0], np.ones(n_sig).T]).T

            z_f = np.max(particle[:,1])
            while(True):
                trns = sigmoidal_fit[:,0]**2 + (sigmoidal_fit[:,1] - z_f)**2
            
                slope, radius = np.linalg.lstsq(temp, trns, rcond=None)[0]

                if slope < 1e-5 or z_f < -4000.0:
                    break

                z_f -= min(slope, r)


            ###STEP 4.4: calculate the WCA###
            r_f = np.sqrt(radius)
                
            d_ = np.sqrt(np.max(particle[:,0]))

            wca = np.arcsin(d_/r_f) + np.pi*0.5

            wca_tot += wca

            if debug:
                print("z_f = ", z_f, ", slope = ", slope, ", r_f = ", r_f)
                print("wca = ", wca*(180.0/np.pi), " [", file_, "]", flush=True)
        
        return wca_tot*(180.0/np.pi)/n_files






########################
# Friction coef. & viscosity
########################
def water_in_cnt(A_eps, B, sigma, rc_cw):
    ##############################
    # Stage 0. Initialize Mirheo
    ##############################
    u = mir.Mirheo(ranks, domain_frct_visc, debug_level=3, log_filename='log', comm_ptr=MPI._addressof(MPI.COMM_WORLD))
    
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

    if debug:
        u.registerPlugins(mir.Plugins.createStats('stats', every=200))
        u.run(2002, dt=dt)
    else:
        u.run(10000, dt=dt)

    u.deregisterIntegrator(minimize)
    del minimize

    ##############################
    # Stage 2. Velocity-Verlet & get temperature right (equilibrium) (should take 0.2ns)
    ##############################
    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv_w)
    

    ####Plugins####				
    u.registerPlugins(mir.Plugins.createBerendsenThermostat('thermostat', [pv_w], T=T, tau=ureg('2.0 ps')))   #(Werder et al. 2003)
    
    if debug:
        u.run(1002, dt=dt)
    else:
        u.run(100000, dt=dt)

    ##############################
    # Stage 3. Save stress tensor & total tangential force with Miheo plugins into csv files
    ##############################
    path = "data/h5_w_in_cnt/"
    dump_every = 1

    ####Viscosity####
    u.registerPlugins(mir.Plugins.createStressTensor('stress_tensor', pv_w, dump_every, path))

    ####Friction coef.####
    u.registerPlugins(mir.Plugins.createTotalForceSaver('tangential_force', pv_w, dump_every, path))

    if debug:
        dump_every = 100
        u.registerPlugins(mir.Plugins.createDumpParticles('part_dump_w', pv_w, dump_every, [], path + 'pv_w-'))

        u.run(1002, dt=dt)
    else:
        u.run(1000000, dt=dt)

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
