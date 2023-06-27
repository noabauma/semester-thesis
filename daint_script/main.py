import numpy as np
import os
import shutil
from mpi4py import MPI

from mirheo_scripts import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def main():
    #jobID  = int(os.getenv('SLURM_ARRAY_JOB_ID'))
    #taskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    jobID = 5
    taskID = 5

    path = "data_" + str(jobID)
    str_taskID = str(taskID)

    os.makedirs(path + "/result/", exist_ok=True)

    data_job_task_id = path + "/" + str_taskID
    
    """
    parameters = np.loadtxt("samples-100.csv", dtype=np.float64, skiprows=1, delimiter=",")

    A_eps_min = 1.0       #[kcal/mol]
    A_eps_max = 2.0
    
    B_min = 0.0            #[1]
    B_max = 2.0
    
    sigma_min = 2.0        #[A]
    sigma_max = 5.0
    
    rc_min = 5.0           #[A]
    rc_max = 15.0
    
    parameters = parameters[taskID,:]
    
    A_eps = (A_eps_max - A_eps_min)*parameters[0] + A_eps_min
    B     = (B_max - B_min)*parameters[1] + B_min
    sigma = (sigma_max - sigma_min)*parameters[2] + sigma_min
    rc    = (rc_max - rc_min)*parameters[3] + rc_min
    """
    parameters = [0.3194, 1.4430, 2.1485, 7.6550]
    A_eps = parameters[0]
    B     = parameters[1]
    sigma = parameters[2]
    rc    = parameters[3]

    wca_grs 	 = droplet_grs(A_eps * 6.947695457055374e-21, B, sigma, rc, data_job_task_id)
    wca_cnt 	 = droplet_cnt(A_eps * 6.947695457055374e-21, B, sigma, rc, data_job_task_id)
    viscosity, frict_coef = water_in_cnt(A_eps * 6.947695457055374e-21, B, sigma, rc, data_job_task_id)

    if rank == 0:
        shutil.rmtree(data_job_task_id, ignore_errors=True)

        fname = path + "/result/" + str_taskID + ".txt"
        result = np.array([taskID, A_eps, B, sigma, rc, wca_grs, wca_cnt, viscosity, frict_coef])
        np.savetxt(fname, result.reshape((1,result.shape[0])), fmt='%.8e', delimiter=',')
    
    

if __name__ == "__main__":
    main()
