import numpy as np
import matplotlib.pyplot as plt

def main():
    parameters = np.loadtxt("samples-1000.csv", delimiter=',', dtype=np.float64, skiprows=1)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(parameters[:,0], parameters[:,1], parameters[:,2])

    plt.show()


if __name__ == "__main__":
    main()