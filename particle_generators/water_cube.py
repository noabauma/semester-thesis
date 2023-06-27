import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

if __name__ == "__main__":

    #parameters
    Lx = 10.0                           #nm
    Ly = 10.0                           #nm
    Lz = 21.9                           #nm

    max_len = max(Lx, Ly, Lz)

    #D = 33.328                          #Density [nm^-3]
    D = 36.0
    d = D**(1.0/3.0)                    #Density in 1D [nm^-1]

    print("Number of particles: ", Lx*Ly*Lz*D)

    x = np.linspace(0.0, max_len, int(d*max_len))

    x_len = x.shape[0]

    particle = np.empty((x_len*x_len*x_len, 3))
    for i in range(x_len):
        for j in range(x_len):
            for k in range(x_len):
                particle[i*x_len*x_len + j*x_len + k, :] = np.array([x[i], x[j], x[k]])

    
    #delete the ones that are out of range
    particle_temp = particle[particle[:,0] <= Lx,:]
    particle_temp = particle_temp[particle_temp[:,1] <= Ly,:]
    particle = particle_temp[particle_temp[:,2] <= Lz,:]

    N = particle.shape[0]
    print("actual Number of particles: ", particle.shape[0])
    
    particle *= 10.0    #nm -> angstrom
        
    np.savetxt("water_cube_{:d}.txt".format(N), np.c_[particle[:,0], particle[:,1], particle[:,2]], fmt=['%10.3f', '%10.3f', '%10.3f'])

    """
    #plot for inspection
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(particle[:,0],particle[:,1],particle[:,2])
    max_len *= 10.0
    ax.set_xlim(0.0,max_len)
    ax.set_ylim(0.0,max_len)
    ax.set_zlim(0.0,max_len)
    plt.show()
    """