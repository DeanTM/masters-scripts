from simulation import *

# # TODO: move Dask stuff to a test script

nolearn_genome = [
    0.,-10.,0.,100.,0.,0.,0.,0.,0.,0.,
    0.,0.,0.,0.,0.,0.,0.,0.,0.,100.,0.
]


# Functions for handling the genomes
@jit(nopython=True)
def get_params_from_genome(genome):
    """Genome must be passed in as a numpy array for the jit."""
    params = genome.copy()
    
    p_theta_unbounded = genome[1]  # \in [1, \infty)
    mu_unbounded = genome[2]  # \in [0, 1]
    tau_theta_unbounded = genome[3]  # \in R^+
    tau_e_unbounded = genome[-2]  # \in R^+
    beta_unbounded = genome[-1]  # \in [0, 1]
    
    p_theta = 1+softplus(p_theta_unbounded)
    mu = sigmoid(mu_unbounded)
    tau_theta = softplus(tau_theta_unbounded)
    tau_e = softplus(tau_e_unbounded)
    beta = sigmoid(beta_unbounded)
    
    params[1] = p_theta
    params[2] = mu
    params[3] = tau_theta
    params[-2] = tau_e
    params[-1] = beta
    return params
    
@jit(nopython=True)
def sigmoid(x):
    return 1./(1+np.exp(-x))

@jit(nopython=True)
def softplus(x):
    return np.log(1+np.exp(x))

def run_repeated_trial(
    W_initial, plasticity_params,
    trial_func, n_runs,
    verbose=False,
    seed=None
):
    W = W_initial
    theta = None
    full_results_dict = dict()
    
    iterable = range(n_runs)
    if verbose:
        from tqdm import tqdm
        iterable = tqdm(iterable)
    for i in iterable:
        results_dict = trial_func(
            W=W,
            theta=theta,
            plasticity_params=plasticity_params,
            seed=seed
        )
        W = results_dict["W"][:, :, -1]
        theta = results_dict["theta"][:, -1]
        if i == 0:
            for k, v in results_dict.items():
                full_results_dict[k] = [v]
        else:
            for k, v in results_dict.items():
                full_results_dict[k].append(v)
    return full_results_dict
        
def get_fitness(
    W_initial, plasticity_params,
    trial_func,
    n_runs,
    n_multiples,
    verbose=False
):
    fitness = 0.0
    for i in range(n_multiples):
        results_dict = run_repeated_trial(
            W_initial=W_initial,
            plasticity_params=plasticity_params,
            trial_func=trial_func,
            n_runs=n_runs,
            verbose=verbose,
            seed=i
        )
#         fitness += np.sum(results_dict['reward'])
        for rew_array in results_dict['reward']:
            if not np.any(np.isnan(rew_array)):
                fitness += np.sum(rew_array)
    fitness = fitness / n_multiples  # get the average across restarts
    return fitness