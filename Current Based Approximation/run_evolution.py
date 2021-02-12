from evolution import *

from functools import partial

from dask.distributed import Client, LocalCluster
from deap import base, creator, tools, algorithms, cma

import argparse
import pickle
from os import path
from datetime import datetime

#region: Parse Arguments
parser = argparse.ArgumentParser(description='Run evolutionary algorithm')
parser.add_argument(
    '--n_runs', type=int,
    default=20, help='number of consecutive trials over which to learn'
    )
parser.add_argument(
    '--n_multiples', type=int,
    default=2, help='number of restarts to average performance over'
    )
parser.add_argument(
    '--coherence', type=float,
    default=0.1, help='coherence score for stimulation inputs to neural units'
    )
parser.add_argument(
    '--task', type=str,
    default='2afc', help='task to run ["2afc" or "xor"]'
    )
parser.add_argument(
    '--use_phi_true', action='store_true',
    help='whether to use the true slow Siegert formula for firing rates'
    )
parser.add_argument(
    '--penalty', type=float,
    default=0.5, help='amount to penalise each trial for numerical instability'
    )
parser.add_argument(
    '--sigma', type=float,
    default=1e-3, help='initial sigma for CMA-ES'
)
parser.add_argument(
    '--lambda_', type=int,
    default=16, help='initial lambda for CMA-ES. Inflated due to noisy fitness'
)


args = parser.parse_args()

#endregion

#region: Checkpointing
checkpoint_folder = 'ea_checkpoints'

fname = f'checkpoint_{datetime.now()}.pkl'
checkpoint_fname = path.join(checkpoint_folder, fname)
#endregion

#region: Debugging
nan_debug = False

debug_folder = 'failed_simulations'
def save_nan_debug(plasticity_params, results_dict):
    for nu_array in results_dict['nu']:
        if np.any(np.isnan(nu_array)):
            # save array
            datetime_suffix = datetime.now()
            savefile = dict(
                plasticity_params=plasticity_params,
                results_dict=results_dict
            )
            # add randomness to avoid collisions
            fname = f'nan_simulation_{datetime_suffix}_{np.random.rand()}.pkl'
            with open(path.join(debug_folder, fname), 'wb') as f:
                pickle.dump(savefile, f)
    return None
#endregion

#region: Set Up Task
n_runs = args.n_runs
n_multiples = args.n_multiples
coherence=[-args.coherence, args.coherence]

w_plus = 1.  # start in unlearned state
w_minus = get_w_minus(w_plus)
W_initial = get_weights(
    w_plus=w_plus,
    w_minus=w_minus
)

trial_params = dict(
    p=p,
    f=f,
    N=N,
    w_plus=w_plus,
    w_minus=w_minus,
    task=args.task,
    coherence=coherence,
    n_runs=n_runs,
    n_multiples=n_multiples
    )
    
use_phi_fitted = not args.use_phi_true
run_trial_func = None
if args.task == '2afc':
    run_trial_func == run_trial_coherence_2afc
elif args.task == 'xor':
    run_trial_func = run_trial_XOR
else:
    raise NotImplementedError(f"Task {args.task} not yet implemented.")
trial_func = partial(
    run_trial_func,
    total_time=2*runtime,
    coherence=coherence,
    use_phi_fitted=use_phi_fitted,
)

penalty = args.penalty
def get_reward_from_results(
    results_dict,
    n_runs=n_runs,
    penalty=penalty
    ):
    # the sooner it fails, the further it is from a safe region
    reward = penalty*(len(results_dict['reward'])-n_runs)
    for rew_array in results_dict['reward']:
            if not np.any(np.isnan(rew_array)):
                reward += np.sum(rew_array)
            else:
                reward -= penalty
    return reward * defaultdt

def plasticity_fitness(
    plasticity_params,
    n_runs, n_multiples,
    trial_func=trial_func,
    W_initial=W_initial,
    seeds=None,
    nan_debug=False
):
    # very hacky, but whatever
    if seeds is None:
        seeds = [None] * n_multiples
    
    all_results = [run_repeated_trial(
        W_initial=W_initial,
        plasticity_params=plasticity_params,
        trial_func=trial_func, n_runs=n_runs,
        verbose=False, seed=seeds[i],
        nan_verbose=True  # see when NaNs occur
        )
        for i in range(n_multiples)]
    fitness = 0.
    for results_dict in all_results:
        if nan_debug:
            save_nan_debug(plasticity_params, results_dict)
        fitness += get_reward_from_results(results_dict)
    return fitness


# seeds are necessary for independent simulations
# so we make it the second positional argument in the 
# fitness function so it works well with client.map
def fitness(genome, seeds, n_runs, n_multiples, nan_debug=False):
    plasticity_params = get_params_from_genome(np.array(genome))
    fitness = plasticity_fitness(
        plasticity_params=plasticity_params,
        n_runs=n_runs,
        n_multiples=n_multiples,
        seeds=seeds,
        nan_debug=nan_debug
        )
    return fitness,
#endregion

#region: Set Up EA
creator.create("FitnessMax", base.Fitness, weights=(1.,))  # maximise reward
creator.create("Individual", list, fitness=creator.FitnessMax)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('avg', np.mean)
stats.register('std', np.std)
stats.register('min', np.min)
stats.register('max', np.max)

# increase initial covariance along no-learning manifold
cov_matrix_initial = np.eye(len(nolearn_genome))
cov_matrix_initial[2] = 1e2
cov_matrix_initial[-2] = 1e2
cov_matrix_initial[-1] = 1e2

# parameters are generally tiny, so choose small sigma
sigma = 1e-3
centroid = nolearn_genome
lambda_EA = 16  # increased because noisy fitness function

strategy = cma.Strategy(
    centroid=centroid,
    sigma=sigma,
    cov_matrix_initial=cov_matrix_initial,
    lambda_=lambda_EA,
    weights="linear"  # noisy, don't want to converge too quickly
)

n_gen = 150
checkpoint_freq = 20

hof = tools.HallOfFame(20)  # hopefully it will find values far apart

toolbox = base.Toolbox()
toolbox.register(
    "evaluate", fitness,
    n_runs=n_runs, n_multiples=n_multiples,
    nan_debug=nan_debug
    )
toolbox.register("generate", strategy.generate, creator.Individual)
toolbox.register("update", strategy.update)
#endregion


if __name__ == '__main__':
    cluster = LocalCluster(n_workers=lambda_EA)
    client = Client(cluster)

    pop = toolbox.generate()
    seeds = np.random.random_integers(
        100*n_multiples*len(pop),
        size=(len(pop), n_multiples)
        )
    
    def dask_map(func, *seqs, **kwargs):
        results_future = client.map(func, *seqs, **kwargs)
        return client.gather(results_future)
    
    def dask_map_fitness(fitness_func, pop, **kwargs):
        results = dask_map(fitness_func, pop, seeds, **kwargs)
        return results

    toolbox.register("map", dask_map_fitness)
    
    if checkpoint_freq is None:
        pop, logbook = algorithms.eaGenerateUpdate(
                toolbox,
                ngen=n_gen,
                stats=stats,
                halloffame=hof,
                verbose=True
            )
    else:
        assert checkpoint_freq > 0
        for i in range(0, n_gen, checkpoint_freq):
            print(f"Running to next checkpoint, starting from gen. {i}")
            pop, logbook = algorithms.eaGenerateUpdate(
                toolbox,
                ngen=checkpoint_freq,
                stats=stats,
                halloffame=hof,
                verbose=True
            )

            state = dict(
                population=[list(x) for x in pop],
                # fitnesses=[x.fitness.values for x in pop],
                halloffame=[list(x) for x in hof],
                # logbook=logbook,
                generation=i + checkpoint_freq,
                covariance=strategy.C,
                trial_params=trial_params
            )
            print("..checkpointing..")
            with open(checkpoint_fname, 'wb') as f:
                pickle.dump(state, f)
        print(f"Running to final section, starting from gen. {i}")
        pop, logbook = algorithms.eaGenerateUpdate(
            toolbox,
            ngen=(n_gen % checkpoint_freq),
            stats=stats,
            halloffame=hof,
            verbose=True
        )

    print("Final Population:\n", *pop)

