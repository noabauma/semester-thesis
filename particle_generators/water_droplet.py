import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

if __name__ == "__main__":

    #parameters
    d = 6.93                           #diameter in nm
    D = 33.328                          #Density nm^-3

    r = d*0.5
    V = 4.0/3.0 * np.pi * r**3
    N = D*V                             #Number of particles

    D_ = int(D*d**3)

    particle = np.random.rand(D_,3)     #generate N random particles in [0,1]^3

    #center particles
    for i in range(D_):
        particle[i,:] = particle[i,:]*d - np.array([r, r, r])
    
    deleteable = []
    for i in range(D_):
        if particle[i,0]**2 + particle[i,1]**2 + particle[i,2]**2 > r*r:
            deleteable.append(i)

    particle = np.delete(particle, deleteable, 0)

    particle *= 10.0    #nm -> angstrom

    D_ = particle.shape[0]
        
    np.savetxt("water_droplet_{:d}.txt".format(int(d*10.)), np.c_[particle[:,0], particle[:,1], particle[:,2]], fmt=['%10.3f', '%10.3f', '%10.3f'])

    print("What I should have: ", N,"\tWhat I actually have: ",D_)
    """
    #plot for inspection
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(particle[:,0],particle[:,1],particle[:,2])
    plt.show()
    """