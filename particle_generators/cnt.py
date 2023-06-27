# Script to generate a numpy array and a .txt file 
# with the coordinates of a carbon nanotube
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import argparse

# C, mass = 12.0107, charge = 0.0
L = 200.0 # length in A
m = 96 # m chirality index
n = 0  # n chirality index
# chiraliy indices determine archair (m,m), zigzag (m,0) or random (m,n) CNT 
CC = 1.42436 # C-C bond length in A (1.418) (1.42436)

water = True   #if wanted to fill the CNT with water

xyz_file = False #create xyz file, else txt file

a = CC # unit cell factor

# greatest common divisor 
num1 = 2*m+n
num2 = 2*n+m
while (num1!=num2):
    if (num1>num2): num1 = num1 - num2
    else: num2 = num2 - num1
d_R = num1

# Compute geometric properties
C = a*np.sqrt(3.*(n*n + m*n + m*m))
R = C/(2.*np.pi)
L_cell = np.sqrt(3.)*C/d_R

print("Radius: ", R, " angstrom")

# fudge radius of the CNT so that bonds are not short
nchord = 2.0*m*np.sqrt(3.*(n*n + m*n + m*m))/np.sqrt(3.*m*m)
rfudge = np.pi/nchord/np.sin(np.pi/nchord)

# Number of unit cells
N_cell = int(np.ceil(L/L_cell))

# index min/max
pmin = 0
pmax = int(np.ceil(n + (n + 2.*m)/d_R))
qmin = int(np.floor(-(2.*n + m)/d_R))
qmax = m

i = 0
coord1 = []
coord2 = []
# generate unit cell coordinates
for q in np.arange(qmin,qmax):
    for p in np.arange(pmin,pmax):
        # first basis atom
        xprime1 = 3.0*a*a*(p*(2.0*n + m) + q*(n + 2.0*m))/(2.0*C)
        yprime1 = 3.0*np.sqrt(3.0)*a*a*(p*m - q*n)/(2.0*C)
        # second basis atom
        xprime2 = xprime1 + 3.0*a*a*(n + m)/(2.0*C)
        yprime2 = yprime1 - a*a*np.sqrt(3.0)*(n - m)/(2.0*C)

        phi1 = xprime1/R
        phi2 = xprime2/R

        if ( (0<=xprime1) and (p*(2.0*n + m) + q*(n + 2.0*m) < 2.0*(n*n + n*m + m*m)) and (0<=yprime1) and (d_R*(p*m-q*n) < 2.0*(n*n+n*m+m*m))):
            coord1.append(np.array([rfudge*R*np.cos(phi1), rfudge*R*np.sin(phi1), yprime1]))
            coord2.append(np.array([rfudge*R*np.cos(phi2), rfudge*R*np.sin(phi2), yprime2]))
            i+=1

Natom = i

# Generate nanotube coordinates
xyzlist = []
for j in range(N_cell):
    for i in range(Natom):
        xyzlist.append(np.array([coord1[i][0], coord1[i][1], coord1[i][2]+j*L_cell]))
        xyzlist.append(np.array([coord2[i][0], coord2[i][1], coord2[i][2]+j*L_cell]))


if (len(xyzlist)!=2*N_cell*Natom): print('Error: Natoms does not match length of xyz list')
Natoms = 2*N_cell*Natom
xyzVec = np.asarray(xyzlist)

#fill CNT with water (Noah Baumann)
if water:
    D = 0.05014                            #Density (normally 0.033328) [A^-3]
    d = D**(1.0/3.0)                    #Density in 1D [A^-1]

    max_len = max(L, 2.0*R)
    x = np.linspace(0.0, max_len, int(d*max_len), endpoint=False)

    x_len = x.shape[0]

    particle = np.empty((x_len*x_len*x_len, 3))
    for i in range(x_len):
        for j in range(x_len):
            for k in range(x_len):
                particle[i*x_len*x_len + j*x_len + k, :] = np.array([x[i], x[j], x[k]])

    #delete the ones that are further away from the tube (only happens if 2R > L)
    particle = particle[particle[:,2] <= L, :]

    #shift particles to the center of the CNT
    shift = np.array([R, R, 0.0])
    for i in range(particle.shape[0]):
        particle[i,:] -= shift
    
    gap = 3.3                     #delete the ones too close to the CNT (actual preferable distance away ~3A)
    R2_ = (R - gap)*(R - gap)
    deleteable = []
    for i in range(particle.shape[0]):
        if particle[i,0]**2 + particle[i,1]**2 > R2_:
            deleteable.append(i)

    particle = np.delete(particle,deleteable,0)

#print(xyzVec.shape)
if water:
    Nwater = particle.shape[0]
    print("#Carbon atoms: ", Natoms)
    print("#Water molecules: ", Nwater)
    print("Water density in CNT: ", Nwater/(np.pi*R2_*L), "[A^-3]")
    f = open('cnt_water_{:d}_{:d}_{:d}_test.xyz'.format(m,n,int(L)),'w')
    f.write(str(Natoms + Nwater)+'\n \n')
    for i in range(Natoms):
        f.write('{:s} {:.3f} {:.3f} {:.3f}\n'.format('C',xyzVec[i,0],xyzVec[i,1],xyzVec[i,2]))
    for i in range(Nwater):
        f.write('{:s} {:.3f} {:.3f} {:.3f}\n'.format('W',particle[i,0],particle[i,1],particle[i,2]))
    f.close()
elif xyz_file:
    f = open('cnt_{:d}_{:d}_{:d}.xyz'.format(m,n,int(L)),'w')
    f.write(str(Natoms)+'\n \n')
    for i in range(Natoms):
        f.write('{:s} {:.3f} {:.3f} {:.3f}\n'.format('C',xyzVec[i,0],xyzVec[i,1],xyzVec[i,2]))
    f.close()
else:
    np.savetxt('cnt_{:d}_{:d}_{:d}.txt'.format(m,n,int(L)), np.c_[xyzVec[:,0], xyzVec[:,1], xyzVec[:,2]], fmt=['%10.3f', '%10.3f', '%10.3f'])

"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyzVec[:,0], xyzVec[:,1], xyzVec[:,2])
if water:
    ax.scatter(particle[:,0], particle[:,1], particle[:,2])
plt.show()
"""