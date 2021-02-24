from evolution import *

from functools import partial

from dask.distributed import Client, LocalCluster
from deap import base, creator, tools, algorithms  #, cma  # cma taken from evolution.py

import argparse

import pickle
from time import time
import json
from os import path, mkdir, cpu_count
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import animation


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
    default=1e-1, help='initial sigma for CMA-ES'
)
parser.add_argument(
    '--lambda_', type=int,
    default=16, help='initial lambda for CMA-ES. Inflated due to noisy fitness'
)
parser.add_argument(
    '--n_gen', type=int,
    default=200, help='number of generations for CMA-ES'
)
parser.add_argument(
    '--checkpoint_freq', type=int,
    default=20, help='frequency with which to save checkpoints (in generations)'
)
parser.add_argument(
    '--hof', type=int,
    default=20, help='number of individuals to store in hall of fame'
)
parser.add_argument(
    '--n_workers', type=int,
    default=cpu_count(), help='number of dask workers, defaults to cpu count'
)
# Determine start
start_group = parser.add_mutually_exclusive_group()
start_group.add_argument(
    '--start_trained', action='store_true',
    help='whether to start with params w_+, w_- as given in Brunel&Wang2001'
)
start_group.add_argument(
    '--w_plus', type=float,
    default=1.0, help='w_+ param to start with. Default (untrained) is 1.'
)
parser.add_argument(
    '--show_anim', action='store_true',
    help='whether to show the animation at the end of the EA'
)

args = parser.parse_args()

script_running_datetime = str(datetime.now()).replace(' ', '_')
folder_suffix = '_'.join([__file__[:-3], script_running_datetime])
folder_prefix = path.join(path.join('experiments', folder_suffix))

imagedir = path.join(folder_prefix, 'images_and_animations')
paramsdir = path.join(folder_prefix, 'parameters')
paramsfile = path.join(paramsdir, 'experiment_parameters.json')
checkpointsdir = path.join(folder_prefix, 'checkpoints')
checkpoint_folder = path.join(folder_prefix, 'ea_checkpoints')
checkpoint_fname = path.join(checkpoint_folder, 'checkpoint.pkl')

n_workers = args.n_workers
if n_workers == 0:
    n_workers == args.lambda_
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
            with open(path.join(debug_folder, fname), 'wb') as fp:
                pickle.dump(savefile, fp)
    return None
#endregion

#region: Set Up Task
task = args.task
if task == '2afc':
    run_trial_func = run_trial_coherence_2afc
    coherence = [-args.coherence, args.coherence]
elif task == 'xor':
    run_trial_func = run_trial_XOR
    coherence = args.coherence
else:
    raise NotImplementedError(f"Task {args.task} not yet implemented.")
n_runs = args.n_runs
n_multiples = args.n_multiples

# start in unlearned state?
w_plus = 2.1 if args.start_trained else 1.
w_minus = get_w_minus(w_plus)  #  get_w_minus(1.) = 1.
W_initial = get_weights(
    w_plus=w_plus,
    w_minus=w_minus
)

    
use_phi_fitted = not args.use_phi_true
trial_func = partial(
    run_trial_func,
    total_time=2*runtime,
    coherence=coherence,
    use_phi_fitted=use_phi_fitted,
)

penalty = args.penalty
get_rewards = partial(
    get_reward_from_results,
    penalty=penalty, n_runs=n_runs
)

def plasticity_fitness(
    plasticity_params,
    n_runs, n_multiples,
    trial_func=trial_func,
    W_initial=W_initial,
    randomstate=random_state_default,
    nan_debug=False
):    
    all_results = [run_repeated_trial(
        W_initial=W_initial,
        plasticity_params=plasticity_params,
        trial_func=trial_func, n_runs=n_runs,
        verbose=False,
        randomstate=randomstate,
        nan_verbose=True  # see when NaNs occur
        )
        for i in range(n_multiples)]
    fitness = 0.
    for results_dict in all_results:
        if nan_debug:
            save_nan_debug(plasticity_params, results_dict)
        fitness += get_rewards(results_dict)
    # return average fitness!
    return fitness / n_multiples

#endregion

#region: Set Up EA
def fitness(genome, n_runs, n_multiples, nan_debug=False):
    plasticity_params = get_params_from_genome(np.array(genome))
    randomstate = genome.randomstate
    fitness = plasticity_fitness(
        plasticity_params=plasticity_params,
        n_runs=n_runs,
        n_multiples=n_multiples,
        randomstate=randomstate,
        nan_debug=nan_debug
        )
    return fitness,

creator.create("FitnessMax", base.Fitness, weights=(1.,))  # maximise reward
creator.create("Individual", Genome, fitness=creator.FitnessMax)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('avg', np.mean)
stats.register('std', np.std)
stats.register('max', np.max)
stats.register('min', np.min)

# increase initial covariance along no-learning manifold
cov_matrix_initial = np.eye(len(nolearn_genome))
# cov_matrix_initial[2] = 1e2
# cov_matrix_initial[-2] = 1e2
# cov_matrix_initial[-1] = 1e2

# parameters are generally tiny, so choose small sigma
sigma = args.sigma
lambda_EA = args.lambda_  # increased because noisy fitness function
centroid = nolearn_genome

n_gen = args.n_gen
checkpoint_freq = args.checkpoint_freq
# strategy = cma.Strategy(
strategy = CMAStrategy(
    centroid=centroid,
    sigma=sigma,
    cov_matrix_initial=cov_matrix_initial,
    lambda_=lambda_EA,
    weights="equal",  # noisy, don't want to converge too quickly
    # custom parameters:
    store_centroids=True,
    store_covariances=True,
    track_fitnesses=True,
    halloffame=tools.HallOfFame(args.hof),
    checkpoint_dir=checkpointsdir,
    checkpoint_every=checkpoint_freq
)


# hof = tools.HallOfFame(args.hof)  # hopefully it will find values far apart

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
    if not path.exists(folder_prefix):
        mkdir(folder_prefix)
    if not path.exists(imagedir):
        mkdir(imagedir)
    if not path.exists(paramsdir):
        mkdir(paramsdir)
    if not path.exists(checkpointsdir):
        mkdir(checkpointsdir)
    experiment_dict = dict(
        script=__file__,
        n_workers=n_workers,
        nan_debug=nan_debug,
        # task=task,
        # coherence=args.coherence,
        # n_runs=n_runs,
        # n_multiples=n_multiples,
        w_plus_initial=w_plus,
        w_minus_initial=w_minus,
        # use_phi_fitted=use_phi_fitted,
        # penalty=penalty,
        # sigma_initial=sigma,
        # lambda_EA=lambda_EA,
        centroid_initial=centroid,
        cov_matrix_initial=cov_matrix_initial.tolist(),
        # n_gen=n_gen,
        # checkpoint_freq=checkpoint_freq,
        input_args=dict(vars(args))
    )
    parameters_dict.update(experiment_dict)
    with open(paramsfile, 'w') as fp:
        json.dump(parameters_dict, fp)

    cluster = LocalCluster(n_workers=n_workers)
    client = Client(cluster)

    def dask_map(func, *seqs, **kwargs):
        results_future = client.map(func, *seqs, **kwargs)
        return client.gather(results_future)

    toolbox.register("map", dask_map)
    
    # if checkpoint_freq <= 0:
    start = time()
    pop, logbook = algorithms.eaGenerateUpdate(
            toolbox,
            ngen=n_gen,
            stats=stats,
            # halloffame=hof,
            verbose=True
        )
    end = time()
    print(f"Total time taken: {end-start:.2f} seconds")
    
    print("Final Population:\n", *pop, sep='\n')

    if strategy.track_fitnesses:
        fig, axes = plt.subplots(figsize=(12, 6))
        # print("max fitnesses", strategy.fitness_max)
        axes.plot(
            np.arange(len(strategy.fitness_max)),
            strategy.fitness_max,
            label='maximum')
        axes.plot(
            np.arange(len(strategy.fitness_min)),
            strategy.fitness_min,
            label='minimum')
        axes.set_title('Fitness Across the Generations')
        axes.set_xlabel('generation')
        axes.set_ylabel('fitness (a.u.)')
        axes.legend()
        plt.savefig(path.join(imagedir, "plasticity_fitnesses.png"))

    if strategy.store_centroids:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax2 = ax.twinx()
        xlabels = param_names_latex
        xticks = np.arange(len(xlabels))
        tau_mask = np.array(['tau' in s for s in xlabels])

        # TODO: put this function in evolution.py??
        # TODO: make different plots for different sized params
        # TODO: add variance to plots
        def anim_func(i, ax=ax,ax2=ax2, cmap=plt.cm.cool):
            ax.clear()
            ax2.clear()
            start_params = get_params_from_genome(
                np.array(strategy.stored_centroids[0]))
            end_params = get_params_from_genome(
                np.array(strategy.stored_centroids[-1]))
            params = get_params_from_genome(
                np.array(strategy.stored_centroids[i]))
            # print(start_params, tau_mask)
            # print(start_params[~tau_mask])
            ax.plot(
                xticks[~tau_mask], start_params[~tau_mask],
                'x',
                # color='orange'
                color="gray",
                alpha=0.5,
                label='start'
                )
            ax.plot(
                xticks[~tau_mask], end_params[~tau_mask],
                'o',
                # color='orange'
                color="black",
                alpha=0.5,
                label='end'
                )
            ax.plot(
                xticks[~tau_mask], params[~tau_mask],
                '.',
                # color='orange'
                color=cmap(i/(n_gen-1))
                )
            ax2.plot(
                xticks[tau_mask], start_params[tau_mask],
                'x',
                # color='orange'
                color="gray",
                alpha=0.5,
                label='start'
                )
            ax2.plot(
                xticks[tau_mask], end_params[tau_mask],
                'x',
                # color='orange'
                color="black",
                alpha=0.5,
                label='end'
                )
            ax2.plot(
                xticks[tau_mask], params[tau_mask],
                '.',
                # color='orange'
                color=cmap(i/(n_gen-1))
                )
            ax.set_title("Evolution of Plasticity Parameters")
            ax.legend()
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels)
            ax.set_ylabel('param strengths (a.u.)')
            ax2.set_ylabel('time constants (seconds)')

        anim = animation.FuncAnimation(
            fig=fig,
            func=anim_func,
            frames=np.arange(n_gen+1)
            )
        anim.save(
            path.join(imagedir, f'parameters_animation.gif'))
        if args.show_anim:
            plt.show()
