from simulation import *

# Extending the evolutionary algorithm for my own purposes
# such as tracking values and checkpointing
# Note: it's still the case that individual scripts will need
# to specify some functionality, like `dask_map`
from deap import cma

#region Classes for the evolutionary algorithm
class Genome(list):
    """
    A genome which has it's own RandomState for parallelisation.

    The .randomstate is assigned upon generation by the CMA-ES.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.randomstate = None


# TODO: implement checkpointing
class CMAStrategy(cma.Strategy):
    def __init__(
        self,
        *args,
        store_centroids=False,
        store_covariances=False,
        track_fitnesses=False,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        # one random state for each individual
        self.randomstates = [RandomState() for i in range(self.lambda_)]
        self.store_centroids = store_centroids
        self.store_covariances = store_covariances
        self.stored_centroids = None
        self.stored_covariances = None
        self.fitness_max = None
        self.fitness_min = None
        self.track_fitnesses = track_fitnesses

        if self.store_centroids:
            self.stored_centroids = [self.centroid]
        if self.store_covariances:
            self.stored_covariances = [self.C]  # I hope this is covariance!!
        if self.track_fitnesses:
            # from these lists we can compute the max and min so far
            self.fitness_max = []
            self.fitness_min = []
            self.fitness_avg = []
            self.fitness_std = []
    
    def generate(self, ind_init):
        population = super().generate(ind_init)
        for i in range(self.lambda_):
            population[i].randomstate = self.randomstates[i]
        return population
    
    def update(self, population):
        super().update(population)
        if self.store_centroids:
            self.stored_centroids.append(self.centroid)
        if self.store_covariances:
            self.stored_covariances.append(self.C)
        if self.track_fitnesses:
            sorted_population = sorted(
                population,
                key=lambda x:x.fitness.values[0]
                )
            fitness_vals = [x.fitness.values[0] for x in population]
            self.fitness_max.append(sorted_population[-1].fitness.values[0])
            self.fitness_min.append(sorted_population[0].fitness.values[0])
            self.fitness_avg.append(np.mean(fitness_vals))
            self.fitness_std.append(np.std(fitness_vals))
#endregion


#region Functions for handling the genomes
@jit(nopython=True)
def sigmoid(x):
    return 1./(1+np.exp(-x))

@jit(nopython=True)
def logit(x):
    return np.log(x / (1-x))

@jit(nopython=True)
def softplus(x):
    return np.log(1+np.exp(x))

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
#endregion


def run_repeated_trial(
    W_initial, plasticity_params,
    trial_func, n_runs,
    verbose=False,
    nan_verbose=False,
    randomstate=random_state_default
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
            randomstate=randomstate
        )
        W = results_dict["W"][:, :, -1]
        theta = results_dict["theta"][:, -1]
        if i == 0:
            for k, v in results_dict.items():
                full_results_dict[k] = [v]
        else:
            for k, v in results_dict.items():
                full_results_dict[k].append(v)
        # stop repeating if weights are NaN
        nu = results_dict["nu"][:, -1]
        if np.any(np.isnan(W)) or np.any(np.isnan(nu)):
            if nan_verbose:
                print(f"NaN encountered in trial {i+1}/{n_runs}")
            break
    return full_results_dict


def get_reward_from_results(
    results_dict,
    n_runs=0,  # used for penalty
    penalty=0.  # subtracted for each numerically failed run
):
    reward = penalty * (len(results_dict['reward'])-n_runs)
    for rew_array in results_dict['reward']:
            if not np.any(np.isnan(rew_array)):
                # scale rewards by timestep size for rectangular 
                # integral of reward trace
                reward += np.sum(rew_array) * defaultdt 
            else:
                reward -= penalty
    return reward 




# # not used in run_evolution.py
# def get_fitness(
#     W_initial, plasticity_params,
#     trial_func,
#     n_runs,
#     n_multiples,
#     verbose=False,
#     randomstate=random_state_default
# ):
#     fitness = 0.0
#     for i in range(n_multiples):
#         results_dict = run_repeated_trial(
#             W_initial=W_initial,
#             plasticity_params=plasticity_params,
#             trial_func=trial_func,
#             n_runs=n_runs,
#             verbose=verbose,
#             randomstate=randomstate
#         )
#         for rew_array in results_dict['reward']:
#             if not np.any(np.isnan(rew_array)):
#                 fitness += np.sum(rew_array)
#     fitness = fitness / n_multiples  # get the average across restarts
#     return fitness
