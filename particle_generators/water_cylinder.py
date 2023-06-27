import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

if __name__ == "__main__":

    #parameters
    D = 33.328                          #Density nm^-3
    L = 10.0                            #length in nm
    R = 3.4696                          #Radius in nm

    L_2 = L*0.5
    R2 = R*R

    N = max(L, 2.0*R)
    N_2 = N*0.5

    D_ = int(D*N**3)

    particle = np.random.rand(D_,3)     #generate N random particles in [0,1]^3

    #center particles
    shift = np.array([N_2, N_2, N_2])
    for i in range(D_):
        particle[i,:] = particle[i,:]*N - shift
    
    deleteable = []
    for i in range(D_):
        if particle[i,0]**2 + particle[i,1]**2 > R2 or particle[i,2] < -L_2 or  particle[i,2] > L_2:
            deleteable.append(i)

    particle = np.delete(particle, deleteable, 0)

    particle *= 10.0    #nm -> angstrom

    D_ = particle.shape[0]
        
    np.savetxt("water_cylinder2_{:d}.txt".format(D_), np.c_[particle[:,0], particle[:,1], particle[:,2]], fmt=['%10.3f', '%10.3f', '%10.3f'])

    print("Water cylinder size: ", D_)
    
    """
    #plot for inspection
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(particle[:,0],particle[:,1],particle[:,2])
    plt.show()
    """