# Script to generate a numpy array and a .txt file 
# with the coordinates of a graphene sheet
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import argparse
# Usage: python3 graphene.py -lx <Lx> -ly <Ly> -type <edge type> -nlayers <number of layers for multilayer graphene>
# C, mass = 12.0107, charge = 0.0
Lx = 50.0 # in A
Ly = 50.0 # in A
CC = 1.42436 # C-C bond length in A (1.418)
armchair = True # if False, type = zigzag
nlayers = 2 # increase for multilayer graphene
layer_distance = 3.35   #distance between layers in A

xyz_file = False #create xyz file, else txt file

a = CC # unit cell factor

# calculate number of unit cells
if (armchair==True):
    Lx_cell = 2.*a*np.sin(60.*np.pi/180.)
    Ly_cell = 3.*a
else:
    Lx_cell = 3.*a
    Ly_cell = 2.*a*np.sin(60.*np.pi/180.)
Nx_cell = int(np.ceil(Lx/Lx_cell))
Ny_cell = int(np.ceil(Ly/Ly_cell))

# Unit cell coordinates
if (armchair==True):
    r1 = np.array([0.,0.,0.])
    r2 = np.array([-a*np.sin(60.*np.pi/180.), a*np.cos(60.*np.pi/180.), 0.])
    r3 = r2 + np.array([0.,a,0.])
    r4 = np.array([0.,2.*a,0.])
    l_shift = np.array([0.,a,0.])
else: # zigzag
    r1 = np.array([0.,0.,0.])
    r2 = np.array([-a*np.cos(60.*np.pi/180.), a*np.sin(60.*np.pi/180.), 0.])
    r3 = np.array([a,0.,0.])
    r4 = r2 + np.array([2.*a,0.,0.])
    l_shift = np.array([a/2.,a*np.sin(60.*np.pi/180.),0.])

# Generate graphene coordinates
xyzlist = []
Natoms = 0
for k in range(nlayers):
    for j in range(Ny_cell):
        for i in range(Nx_cell):
            r_shift = np.array([i*Lx_cell, j*Ly_cell, k*layer_distance])
            if (k%2!=0): r_shift = r_shift + l_shift
            xyzlist.append(r1+r_shift)
            xyzlist.append(r2+r_shift)
            xyzlist.append(r3+r_shift)
            xyzlist.append(r4+r_shift)
            Natoms += 4

if (len(xyzlist)!=Natoms): print('Error: Natoms does not match length of xyz list')
xyzVec = np.asarray(xyzlist)
strs = ['C' for i in range(Natoms)]
#print(xyzVec.shape)
if xyz_file:
    f = open('grs_{:d}_{:d}.xyz'.format(int(Lx),int(Ly)),'w')
    f.write(str(Natoms)+'\n \n')
    for i in range(Natoms):
        f.write('{:s} {:.3f} {:.3f} {:.3f}\n'.format('C',xyzVec[i,0],xyzVec[i,1],xyzVec[i,2]))
    f.close()
else:
    np.savetxt('grs_{:d}_{:d}_{:d}.txt'.format(nlayers, int(Lx),int(Ly)), np.c_[xyzVec[:,0], xyzVec[:,1], xyzVec[:,2]], fmt=['%10.3f', '%10.3f', '%10.3f'])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyzVec[:,0], xyzVec[:,1], xyzVec[:,2])
plt.show()
