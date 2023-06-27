import numpy as np
import matplotlib.pyplot as plt
import h5py

#this code calculates the density profile of a CNT tube filled with water
if __name__ == "__main__":
    converter = 18.015/0.602214076  #converts [A^-3] to [g/cm^3]

    #load particles (normaly from h5 file)
    load_h5 = False
    if load_h5:
        #water = h5py.File("../other/h5_water_droplet/pv-00002.h5", "r")["position"][()]
        #water *= 10.0 #nm -> A
        water = h5py.File("../other/h5_cnt_water2/pv_w-00035.h5", "r")["position"][()]
    else:
        water = np.loadtxt("../particle_generators/water_droplet_200.txt", dtype=np.float64)

    Nwater = water.shape[0]

    center = np.array([np.mean(water[:,0]), np.mean(water[:,1]), np.mean(water[:,2])])  #center point of the cnt
    water -= center

    water_r2 = np.sqrt(water[:,0]**2 + water[:,1]**2 + water[:,2]**2)

    R = (np.max(water[:,0]) - np.min(water[:,0]) + np.max(water[:,1]) - np.min(water[:,1]) + np.max(water[:,2]) - np.min(water[:,2]))/6.0


    print("Radius: ", R)
    print("Density: ", Nwater/((4.0/3.0)*np.pi*R*R*R))
    
    n_binning = 400                                 #number of times we bin in radial direction
    radiuses = np.linspace(0.0, R*1.1, n_binning)
    binning_depth = radiuses[1]                     #range [angstrom] we count particles [r - binning_depth/2, r + binning_depth/2]

    r = np.empty((n_binning, 4), dtype=np.float64)
    for i in range(n_binning):
        r_ = radiuses[i]
        r[i,0] = (r_)
        r[i,1] = (r_ + binning_depth)
        r[i,2] = 1.0/((4.0/3.0)*np.pi*(r[i,1]**3 - r[i,0]**3))

    rho = np.empty((n_binning), dtype=np.float64)
    for j in range(n_binning):
        
        count = np.count_nonzero((r[j,0] <= water_r2) & (water_r2 < r[j,1]))
        
        rho[j] = float(count)*r[j,2]


    #plot
    plt.plot(radiuses[:], rho[:]*converter, '.-', label='bins')
    plt.plot(radiuses[:], np.full((n_binning,1), 0.033328*converter), 'r--', label='{:.3f}'.format(0.033328*converter) + r'$\:[g/cm^3]$')
    
    plt.title("Radial density profile of a water droplet")
    plt.xlabel(r'$radius\: r \: [\AA]$')
    plt.ylabel(r'$density\: \rho \: [g/cm^3]$')
    #plt.ylim(0.0, 3.0)
    plt.legend(loc='best')
    plt.grid()
    plt.show()
