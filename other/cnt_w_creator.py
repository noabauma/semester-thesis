import numpy as np
import os
import glob
import h5py

#this script will only be used ones to delete the water molecules outside the cnt
#and save the water particles into a new h5 file

if __name__ == "__main__":
    #load last save h5 file
    water_files = glob.glob("h5_cnt_water2/pv_w-*.h5")
    water_files = np.sort(water_files)
    w_n_files = len(water_files)
    w_pos = h5py.File(water_files[w_n_files-1], "r")["position"][()]
    w_vel = h5py.File(water_files[w_n_files-1], "r")["velocity"][()]

    cnt_files = glob.glob("h5_cnt_water2/pv_c-*.h5")
    cnt_files = np.sort(cnt_files)
    c_n_files = len(cnt_files)
    c_pos = h5py.File(cnt_files[c_n_files-1], "r")["position"][()]

    center = np.array([np.mean(c_pos[:,0]), np.mean(c_pos[:,1]), np.mean(c_pos[:,2])])  #center point of the cnt

    #R2_cnt = ((np.max(c_pos[:,0]) - np.min(c_pos[:,0]) + np.max(c_pos[:,1]) - np.min(c_pos[:,1]))*0.25)**2
    R2_cnt = 0.0
    c_n = c_pos.shape[0]
    for i in range(c_pos.shape[0]):
        R2_cnt += (center[0] - c_pos[i,0])**2 + (center[1] - c_pos[i,1])**2
    R2_cnt /= float(c_n)
    print("cnt R2 = ", R2_cnt)

    n = w_pos.shape[0]
    print("Before n = ",  n)

    w_R2_pos = np.empty((n), dtype=np.float64)
    for i in range(n):
        w_R2_pos[i] = (center[0] - w_pos[i,0])**2 + (center[1] - w_pos[i,1])**2

    c_pos_z = c_pos[:,2]
    w_pos_z = w_pos[:,2]
    
    keep = (w_R2_pos < R2_cnt) & (w_pos_z <= np.max(c_pos_z)) & (np.min(c_pos_z) <= w_pos_z)

    w_pos = w_pos[keep]
    w_vel = w_vel[keep]

    n = w_pos.shape[0]
    print("After n=",  n)

    id = np.reshape(np.arange(n), (n,1))

    #create new restart files for mirheo
    os.remove('../particle_generators/water_for_cnt_96_0_200_2/pv.PV-00000.h5')
    f = h5py.File('../particle_generators/water_for_cnt_96_0_200_2/pv.PV-00000.h5', 'a')

    dset = f.create_dataset('position', (n,3), dtype=np.float64)
    dset[...] = w_pos
    dset = f.create_dataset('velocity', (n,3), dtype=np.float64)
    dset[...] = w_vel
    dset = f.create_dataset('id', (n,1), dtype=np.int)
    dset[...] = id
    
    f.close()