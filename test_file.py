# this is just a test file to run various components of the algorithms
# and parallelisation etc. from.
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from tqdm import tqdm
# from dask.distributed import LocalCluster, Client

# dt = 0.5e-3
# tau_vals = np.array((1e-3, 0.5e-2, 1e-2))
# def dydt(y, tau):
#     return -y/tau


def dask_func(args):
    return args**2


if __name__ == '__main__':
    # y_vals = np.zeros((50, tau_vals.shape[0]))
    # y_vals[0, :] = 1./tau_vals
    # for i in range(1, y_vals.shape[0]):
    #     y_vals[i,:] = y_vals[i-1, :] + dt*dydt(y_vals[i-1,:], tau=tau_vals)
    
    # for j in range(y_vals.shape[1]):
    #     plt.plot(y_vals[:,j], label=f"{tau_vals[j]}: {y_vals[:,j].sum()*dt:.3f}")
    # plt.legend()
    # plt.show()

    # cluster = LocalCluster()
    # client = Client(cluster)


    # futures = client.map(dask_func, np.arange(100))
    # print(client.gather(futures))

    total_time = 1e-2
    times = np.linspace(0., total_time, int(total_time/defaultdt))
    samples = 10000
    total_increments = []
    for i in tqdm(range(samples)):
        sample = np.sum([dic_noise_dt(np.array(0.))*defaultdt for t in times])
        total_increments.append(sample)
    plt.hist(total_increments, bins=100)
    plt.title(f'Noise increments over {total_time}\ndt={defaultdt}')
    plt.savefig(f'noise_increments_dt_{defaultdt}.png')

    
