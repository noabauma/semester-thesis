import numpy as np
from scipy.optimize import curve_fit
import h5py
import glob

###
#This script will calculate the contact angle of water droplets on graphene sheets. (Pls use #water_molecule > 16000 for more accuracy)
###

#sigmoidal curve for fitting the density function
def sigmoidal(z, rho_l, z_e, d):
    #rho_l = 0.033328        #density of water [0.0167, 0.0368]
    return rho_l*0.5*(1.0 - np.tanh(2.0*(z - z_e)/d))


#load particle (either in debug mode or h5file)
def load_particles(debug, i):
    if debug:
        ###manipulate the droplet for debugging###
        particles = np.loadtxt("../particle_generators/water_droplet_129.txt")

        n = particles.shape[0]

        max_z = np.max(particles[:,2])
        r = (np.max(particles[:,0]) - np.min(particles[:,0]) + np.max(particles[:,1]) - np.min(particles[:,1]) ) * 0.25
        cut_per = .5   #how much to be cut [0.0 = all gone, 1.0 = all stay]
        cut = r*cut_per
        cut2 = cut*cut
        deleteable = []
        for i in range(n):
            if particles[i,0]**2 + particles[i,1]**2 > cut2:
                deleteable.append(i)
        particles = np.delete(particles, deleteable, 0)

        h_2 = (np.max(particles[:,2]) - np.min(particles[:,2]))*0.5

        true_wca = np.arcsin(cut/h_2) + np.pi*0.5
        print("What I actually should get = ", true_wca*(180.0/np.pi))
        ##########################################
    else:
        all_files = glob.glob("../h5_droplet_cnt/pv_w-*.h5")
        all_files = np.sort(all_files)
        n_files = len(all_files)
        particles = h5py.File(all_files[n_files-1-i], "r")["position"][()]
    
    return particles



if __name__ == "__main__":
    debug = False
    
    wca_tot = 0.0

    runs = 10 if not debug else 1
    for i in range(runs):
        particles = load_particles(debug, i)
        
        n = particles.shape[0]

        print("#particles", n)

        particles_x = particles[:,0]
        particles_y = particles[:,1]
        particles_z = particles[:,2]

        d = (np.max(particles_x) - np.min(particles_x) + np.max(particles_y) - np.min(particles_y) ) * 0.5
        d_ = d*0.5                                      #Radius from Werder et al. 2001
        h = np.max(particles_z) - np.min(particles_z)
        
        mean_x = np.mean(particles_x)
        mean_y = np.mean(particles_y)
        mean_z = np.mean(particles_z)

        #shift droplet to the center
        particle = np.empty((n,2), dtype=np.float64)
        shift = np.array([mean_x, mean_y, mean_z])
        for i in range(n):
            particles[i,:] -= shift
            particle[i,0] = particles[i,0]*particles[i,0] + particles[i,1]*particles[i,1]
            particle[i,1] = abs(particles[i,2])  
        

        #bins properties
        h_bins = 10.0                #height of bins in angstrom
        dA = 95.0                    #base area per bin angstrom^2
        r_bins = np.sqrt(dA/np.pi)   #bin radius
        r_bins_half = r_bins*0.5
        r_bins_2 = r_bins*r_bins

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



        ###STEP 1: calculate all the M_bins (xy-direction) density profiles###
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
        


        ###STEP 2: sigmoidal curve fit###
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



        ###STEP 3: Fit circular curve (find z_f and r_f)###
        sigmoidal_fit = np.array(sigmoidal_fit)
        n = sigmoidal_fit.shape[0]
        temp = np.vstack([sigmoidal_fit[:,0], np.ones(n).T]).T

        z_f = np.max(particle[:,1])
        slope_old = -1.0
        while(True):
            trns = sigmoidal_fit[:,0]**2 + (sigmoidal_fit[:,1] - z_f)**2
        
            slope, radius = np.linalg.lstsq(temp, trns, rcond=None)[0]

            #print("[", slope1, ",", radius1, end="]\n", flush=True)
            if abs(slope_old - slope) < 1e-5:
                break

            slope_old = slope
            #if slope < 0.0 or z_f < -2000.0:
            #    break

            z_f -= min(slope, r)


        r_f = np.sqrt(radius)

        print("z_f = ", z_f, ", slope = ", slope, ", r_f = ", r_f)
            
        d_ = np.sqrt(np.max(particle[:,0]))

        wca = np.arcsin(d_/r_f) + np.pi*0.5

        print("WCA = ", wca, " in degree '", wca*(180.0/np.pi), "'")

        wca_tot += wca*(180.0/np.pi)
    
    print("Average degree: ", wca_tot/runs)
