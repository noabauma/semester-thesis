from scipy.stats import maxwell
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

if __name__ == "__main__":
    a = np.sqrt(1.38e-23*300/(0.018015/6.02214076e23))
    
    fig, ax = plt.subplots(1, 1)
    #r = maxwell.rvs(size=(3000,3), scale=a, random_state=42)
    r = np.random.normal(size=(10000,3), scale=a)

    print("mean: ", np.mean(r))
    n = r.shape[0]
    print(r)
    r_ = 0.0
    for i in range(r.shape[0]):
        r_ += np.sqrt(r[i,0]**2 + r[i,2]**2 + r[i,2]**2) 
    print("avg speed: ", r_/n)

    # ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
    # ax.legend(loc='best', frameon=False)
    # plt.show()