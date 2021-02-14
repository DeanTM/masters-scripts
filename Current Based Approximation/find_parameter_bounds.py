import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

from os import path, listdir

from simulation import F_full

dirname = 'failed_simulations'

fnames = [path.join(dirname, fname) for fname in listdir(dirname)]

lag_before_NaN = 5



# for fname in fnames:
#     with open(fname, 'rb') as f:
#         failed_sim = pickle.load(f)
#     break

# plasticity_params = failed_sim['plasticity_params']

# num_points = 100
# max_val = 30.

# theta = np.ones((1,2)) * 20.
# W = np.array([[0., 1.],[1., 0.]])

# f_vals = np.zeros(num_points)

# for i, nu in enumerate(
#     np.vstack([np.ones(num_points)*19, np.linspace(0, max_val, num_points)]).T
#     ):
#     nu = nu.reshape(-1,1)
#     f_vals[i] = F_full.py_func(nu, W, theta, plasticity_params)[0, 1]

# plt.plot(f_vals)
# plt.show()


def get_nan_index(results_dict):
    nan_index = None
    for k, v in results_dict.items():
        if k in ['times', 'theta', 'reward']: continue  # theta starts with a nan
        for arr in v:
            if k in ['W', 'e']:
                nan_mask = np.max(np.isnan(arr), axis=(0,1))
                # print(k, nan_mask.shape, arr.shape)
            else:
                nan_mask = np.max(np.isnan(arr), axis=0)
                # print(k, nan_mask.shape, arr.shape)
            if nan_mask.max():
                this_nan_index = np.arange(nan_mask.shape[0])[nan_mask].min()
                if nan_index is None:
                    nan_index = this_nan_index
                elif this_nan_index < nan_index:
                    nan_index = this_nan_index
    return nan_index
                    


variable = 'W'
vars_before_nan = []
for fname in fnames:
    with open(fname, 'rb') as f:
        failed_sim = pickle.load(f)
    results_dict = failed_sim['results_dict']
    nan_index = get_nan_index(results_dict)
    var_arrays = results_dict[variable]
    final_var_array = var_arrays[-1]

    idx = nan_index-lag_before_NaN
    if idx > 0:
        # weights_before_nan.append(final_weights_array[:,:,idx])
        if variable in ['W', 'e', 'theta']:
            vars_before_nan.append(final_var_array[:,:,idx])
        # if variable == 'reward':
        #     print(np.shape(final_var_array))
        #     vars_before_nan.append(final_var_array[:, idx])
        else:
            vars_before_nan.append(final_var_array[:,idx])
    elif len(var_arrays) > 1:
        # weights_before_nan.append(weights_arrays[-2][:,:,idx])
        if variable in ['W', 'e', 'theta']:
            vars_before_nan.append(var_arrays[-2][:,:,idx])
        # if variable == 'reward':
        #     vars_before_nan.append(var_arrays[-2][idx])
        else:
            vars_before_nan.append(var_arrays[-2][:,idx])

if variable == 'reward':
    r_arr = np.array(vars_before_nan)
    # print(np.abs(r_arr).max())
    plt.hist(
        r_arr, bins=max(10, int(r_arr.shape[0]/5)),
        # density=True
        )
elif variable == 'W':
    small_counts = [np.sum((x.ravel() < 0.1).astype(int)) for x in vars_before_nan]
    big_counts = [np.sum((x.ravel() > 3.4).astype(int)) for x in vars_before_nan]
    plt.scatter(small_counts, big_counts)
else:
    per_unit_final_values = list(zip(*[x.ravel() for x in vars_before_nan]))
    plt.violinplot(per_unit_final_values)
    for i, values in enumerate(vars_before_nan):
        plt.plot(values.ravel(), 'k', alpha=0.01)
plt.show()