import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob

#water = h5py.File("../particle_generators/water_for_cnt_96_0_200/pv.PV-00000.h5", "r")["position"][()]  #presaved first try (little bit smaller than 0.997 g/cm^3)
#water = h5py.File("../other/h5_cnt_water2/pv_w-00035.h5", "r")["position"][()]                          #other try with little gap (but delete the ones outside)
#water = h5py.File("../h5_w_in_cnt/pv_w-00120.h5", "r")["position"][()]                                   #

#this code calculates the density profile of a CNT tube filled with water
if __name__ == "__main__":
    converter = 18.015/0.602214076  #converts [A^-3] to [g/cm^3]

    #load particles (normaly from h5 file)
    cnt   = h5py.File("../particle_generators/h5_cnt_96_0_200/pv_c-00001.h5", "r")["position"][()]

    Ncnt = cnt.shape[0]

    center = np.array([np.mean(cnt[:,0]), np.mean(cnt[:,1]), np.mean(cnt[:,2])])  #center point of the cnt
    cnt   -= center

    R = (np.max(cnt[:,0]) - np.min(cnt[:,0]) + np.max(cnt[:,1]) - np.min(cnt[:,1])) * 0.25   #radius of the CNT
    L = np.ceil(np.max(cnt[:,2]) - np.min(cnt[:,2]))                                         #CNT length

    #bins properties
    n_binning = 400         #number of times we bin in radial direction
    radiuses = np.linspace(0.0, R, n_binning)
    binning_depth = radiuses[1]
    rho = np.zeros((n_binning), dtype=np.float64)

    #all_files = glob.glob("../h5_w_in_cnt/pv_w-*.h5")
    #all_files = glob.glob("../h5_w_in_cnt11/pv_w-*.h5")
    all_files = glob.glob("../particle_generators/water_for_cnt_96_0_200/pv.PV-00000.h5")
    all_files = np.sort(all_files)
    n_files   = len(all_files)

    water = h5py.File(all_files[n_files-1], "r")["position"][()]
    Nwater = water.shape[0]

    print("#water particles = ", Nwater)
    print("CNT Radius: ", R)
    print("Density in CNT: ", Nwater/(np.pi*(R-0.0)*(R-0.0)*L), "\n")

    rep = n_files               #don't take the first 10 because they are from minimization
    for i in range(rep):
        water = h5py.File(all_files[n_files-1-i], "r")["position"][()]

        center = np.array([np.mean(water[:,0]), np.mean(water[:,1]), np.mean(water[:,2])])  #center point of the CNT
        #center = np.array([200.0,200.0,110.0])
        water -= center

        water_r2 = water[:,0]**2 + water[:,1]**2

        Nwater = water_r2.shape[0]

        R_w = (np.max(water[:,0]) - np.min(water[:,0]) + np.max(water[:,1]) - np.min(water[:,1])) * 0.25

        ###extra: keep water particles inside the CNT (only use this if needed)
        if True:
            keep = (water_r2 < R*R) & (water[:,2] < np.max(cnt[:,2])) & (np.min(cnt[:,2]) < water[:,2])
            water_r2 = water_r2[keep]
            print("Befor Nwater = ", Nwater)
            Nwater = water_r2.shape[0]
            print("After Nwater = ", Nwater)
        ###

        print("c-w distance: ", R - R_w)
        print("Water density R_w in CNT: ", Nwater/(np.pi*(R-3)*(R-3)*L))

        r = np.empty((n_binning, 4), dtype=np.float64)
        for i in range(n_binning):
            r_ = radiuses[i]
            r[i,0] = r_*r_
            r[i,1] = (r_ + binning_depth)*(r_ + binning_depth)
            r[i,2] = 1.0/(L*np.pi*(r[i,1] - r[i,0]))

        for j in range(n_binning):
            
            count = np.count_nonzero((r[j,0] <= water_r2) & (water_r2 <= r[j,1]))
            
            rho[j] += float(count)*r[j,2]

        

    rho /= rep  #average over many density profiles

    print("Mean density of the bins: ", np.mean(rho[radiuses <= 35.0]), "\n")
    print("Mean density of the bins: ", np.mean(rho), "\n")
    print("Mean density of the bins: ", np.mean(rho[radiuses <= 33.5]), "\n")

    #plot
    idx = 0
    for i in range(n_binning):
        if radiuses[i] >= 0.0:  #get only the ones after specific radius
            idx = i
            break
    
    rho *= converter

    plt.plot(radiuses[idx:], np.full(radiuses[idx:].shape, 0.033328*converter), 'r--', label='{:.3f}'.format(0.033328*converter) + r'$\:[g/cm^3]$')
    plt.plot(radiuses[idx:], rho[idx:], '.-', label='bins')

    plt.vlines(R, ymin=0.0, ymax=np.max(rho[idx:]), color="#000000")
    plt.text(R, np.max(rho[idx:]), 'CNT')
    
    #plt.title("Radial density profile of water inside a CNT")
    plt.xlabel(r'$r \: [\AA]$')
    plt.ylabel(r'$\rho \: [g/cm^3]$')
    #plt.ylim(0.0, 3.0)
    plt.legend(loc='best')
    plt.grid()
    plt.show()
