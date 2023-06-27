import numpy as np
from scipy.optimize import curve_fit
import h5py
import glob

###
#This script will calculate the contact angle of water droplets on graphene sheets. (Pls use #water_molecule > 16000 for more accuracy)
#
#This is a modified version of my original implementation. This time the spacing between the bins defines the sizes of the bins.
#So no overlapping/smoothing. It does not perform as good, because fitting the sigmoidal will get rougher.
###

#sigmoidal curve for fitting the density function
def sigmoidal(z, rho_l, z_e, d):
    #rho_l = 0.033328        #density of water [0.0167, 0.0368]
    return rho_l*0.5*(1.0 - np.tanh(2.0*(z - z_e)/d))


#load particle (either in debug mode or h5file)
def load_particles(debug):
    if debug:
        ###manipulate the droplet for debugging###
        particles = np.loadtxt("../particle_generators/water_droplet_129.txt")

        n = particles.shape[0]

        max_z = np.max(particles[:,2])
        h = np.max(particles[:,2]) - np.min(particles[:,2])
        cut_per = .45   #how much to be cut [0.0 = all gone, 1.0 = all stay]
        cut = h*cut_per
        deleteable = []
        for i in range(n):
            if particles[i,2] < max_z - cut:
                deleteable.append(i)
        particles = np.delete(particles, deleteable, 0)

        r = h*0.5
        true_wca = np.arcsin((r - h*(1.0 - cut_per) + c_w)/r) + np.pi*0.5 if cut_per >= 0.5 else np.pi*0.5 - np.arcsin((r - h*cut_per + c_w)/r)
        print("What I actually should get = ", true_wca*(180.0/np.pi))
        ##########################################
    else:
        all_files = glob.glob("../h5_droplet_grs/pv_w-*.h5")
        all_files = np.sort(all_files)
        n_files = len(all_files)
        particles = h5py.File(all_files[n_files-1], "r")["position"][()]

    return particles



if __name__ == "__main__":
    debug = True

    c_w = 3.0   #bond distance between water and graphene sheet
    
    ###STEP 0: load particles in either debug mode or from h5file###
    particles = load_particles(debug)
    
    n = particles.shape[0]

    print("#particles", n)

    particles_x = particles[:,0]
    particles_y = particles[:,1]
    particles_z = particles[:,2]


    d = (np.max(particles_x) - np.min(particles_x) + np.max(particles_y) - np.min(particles_y) ) * 0.5
    r = d*0.5
    
    mean_x = np.mean(particles_x)
    mean_y = np.mean(particles_y)
    mean_z = np.mean(particles_z)

    
    ###shift droplet to the center###
    min_z = np.min(particles_z)
    shift = np.array([mean_x, mean_y, min_z])
    particle = np.empty((n,2), dtype=np.float64)             #particle container which holds [:,0]=xy-l2-distance & [:,1]=z-coordinate 
    for i in range(n):
        particles[i,:] -= shift
        particle[i,0] = particles[i,0]*particles[i,0] + particles[i,1]*particles[i,1]
        particle[i,1] = particles[i,2]
    
    

    ###bins properties###
    M_bins = 100                                             #number of bins in z-direction  [the more the better but also slower]
    bins_step = np.linspace(0.0, np.max(particle[:,1])*1.1, M_bins)
    h_bins = bins_step[1]

    N_bins = 100                                             #number of bins in xy-direction [the more the better but also slower]
    bins_loc = np.linspace(0.0, r*0.80, N_bins)              #0.50-0.80 are good values
    r_bins = bins_loc[1]
    
    z_0 = 8.0                                                #Cut-off [angstrom] any points smaller than z_0 will be neglected
    z_0_loc = 0                                              #location where z_0 begins
    for j in range(M_bins):
        if bins_step[j] >= z_0:
            z_0_loc = j
            break
    

    r2 = np.empty((N_bins,4), dtype=np.float64)
    for i in range(N_bins):
        r = bins_loc[i]
        r2[i,0] = r*r
        r2[i,1] = (r + r_bins)*(r + r_bins)
        r2[i,2] = 1.0/(h_bins*np.pi*(r2[i,1] - r2[i,0]))



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
    trns = np.empty((n,1), dtype=np.float64)
    temp = np.vstack([sigmoidal_fit[:,0], np.ones(n).T]).T

    z_f = d
    while(True):
        for i in range(n):
            trns[i] = sigmoidal_fit[i,0]**2 + (sigmoidal_fit[i,1] - z_f)**2
    
        slope, radius = np.linalg.lstsq(temp, trns, rcond=None)[0]

        #print("[", slope1, ",", radius1, end="]\n", flush=True)
        if slope < 0.0:
            break

        z_f -= min(slope, r)
        #z_f -= 0.5



    ###STEP 4: calculate the WCA###
    z_f += c_w                  #spacing between w-c
    r_f = np.sqrt(radius)
        
    wca = np.arcsin(z_f/r_f) + np.pi*0.5 if z_f > 0.0 else np.pi*0.5 - np.arcsin(abs(z_f)/r_f)

    print("z_f = ", z_f, ", slope = ", slope, ", r_f = ", r_f)
    print("WCA = ", wca, " in degree '", wca*(180.0/np.pi), "'")
    