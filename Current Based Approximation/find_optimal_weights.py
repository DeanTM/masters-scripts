from evolution import *

from functools import partial

from dask.distributed import Client, LocalCluster
from deap import base, creator, tools, algorithms, cma

from time import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import animation

from os import path, mkdir, cpu_count
import json

import argparse

np.seterr(all='raise')

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
parser.add_argument(
    '--show_anim', action='store_true',
    help='whether to show the animation at the end of the EA'
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

args = parser.parse_args()

script_running_datetime = str(datetime.now()).replace(' ', '_')
folder_suffix = '_'.join([__file__[:-3], script_running_datetime])
folder_prefix = path.join(path.join('experiments', folder_suffix))

imagedir = path.join(folder_prefix, 'images_and_animations')
paramsdir = path.join(folder_prefix, 'parameters')
paramsfile = path.join(paramsdir, 'experiment_parameters.json')
checkpointsdir = path.join(folder_prefix, 'checkpoints')
resultsdir = path.join(folder_prefix, 'results')





# W_opt = [[0.824, 1.017, 0.805, 1.   ],
#         [0.981, 2.081, 0.772, 1.   ],
#         [0.784, 0.958, 2.113, 1.   ],
#         [0.931, 0.92,  1.073, 1.   ]]

n_runs = args.n_runs
n_multiples = args.n_multiples
coherence = args.coherence
penalty = args.penalty

n_workers = args.n_workers
checkpoint_freq = args.checkpoint_freq

n_gen = args.n_gen
n_hof = args.hof
lambda_EA = args.lambda_
sigma_EA = args.sigma

task = args.task
run_trial_func = run_trial_coherence_2afc if task == '2afc' else run_trial_XOR
coherence = [-coherence, coherence] if task == '2afc' else coherence


trial_func = partial(
    run_trial_func,
    total_time=2*runtime,
    coherence=coherence,
    use_phi_fitted=not args.use_phi_true,
    plasticity=False,
)

get_rewards = partial(
    get_reward_from_results,
    penalty=penalty, n_runs=n_runs
)


@jit(nopython=True)
def bound_weights(W):
    return w_max_default * sigmoid(W)

@jit(nopython=True)
def bound_weights_inverse(W):
    return logit(W/w_max_default)

if args.start_trained:
    w_plus = 2.1
else:
    w_plus = args.w_plus
w_minus = get_w_minus(w_plus)

unit_unbound = bound_weights_inverse(1.)
w_plus_unbound = bound_weights_inverse(w_plus)
w_minus_unbound = bound_weights_inverse(w_minus)

w_minus = get_w_minus(w_plus=w_minus)
W_initial = get_weights(
    w_plus=w_plus,
    w_minus=w_minus
)


def weight_fitness(
    W,
    # n_runs=n_runs,
    # n_multiples=n_multiples,
    # trial_func=trial_func,
    randomstate=random_state_default
):
    all_results = [run_repeated_trial(
        W_initial=W,
        plasticity_params=nolearn_parameters,
        trial_func=trial_func,
        n_runs=n_runs,
        nan_verbose=True,
        randomstate=randomstate
        )
        for i in range(n_multiples)]
    fitness = 0.
    for results_dict in all_results:
        fitness += get_rewards(results_dict)
    return fitness / n_multiples

def get_weights_from_genome(genome):
    genome_reshaped = np.array(genome).reshape(p+2,p+1)
    bound_genome = bound_weights(genome_reshaped)
    return np.hstack([bound_genome, W_initial[:,-1].reshape(-1,1)])
    


def fitness(
    genome,
    # n_runs=n_runs,
    # n_multiples=n_multiples
):
    W = get_weights_from_genome(genome)
    randomstate = genome.randomstate
    fitness = weight_fitness(
        W=W,
        randomstate=randomstate
        )
    return fitness,


creator.create("FitnessMax", base.Fitness, weights=(1.,))
creator.create("Individual", Genome, fitness=creator.FitnessMax)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats_weights_max = tools.Statistics(lambda gen: get_weights_from_genome(gen).max())
stats_weights_min = tools.Statistics(lambda gen: get_weights_from_genome(gen).min())
stats.register('avg', np.mean)
stats.register('std', np.std)
stats.register('max', np.max)
stats.register('min', np.min)
# mstats = tools.MultiStatistics(fitnesses=stats, w_max=stats_weights_max, w_min=stats_weights_min)
# mstats.register('max', np.max)
# mstats.register('min', np.min)

# centroid = list(W_initial.ravel())
# centroid = [-0.18232156 for i in range(W_initial.ravel().shape[0])]
centroid = np.full_like(W_initial[:, :-1], unit_unbound)
centroid[W_initial[:, :-1] == w_plus] = w_plus_unbound
centroid[W_initial[:, :-1] == w_minus] = w_minus_unbound
centroid = [x for x in centroid.ravel()]  # don't have plastic weights change

strategy = CMAStrategy(
    centroid=centroid,
    lambda_=lambda_EA,
    sigma=sigma_EA,
    store_centroids=True,
    store_covariances=True,
    track_fitnesses=True,
    halloffame=tools.HallOfFame(n_hof),
    checkpoint_every=checkpoint_freq,
    checkpoint_dir=checkpointsdir,
    # weights='linear'  # don't want to overemphasize chance
    weights='equal',  # TODO: add to arguments
    
)

# hof = tools.HallOfFame(n_hof)

toolbox = base.Toolbox()
toolbox.register("evaluate", fitness)
toolbox.register("generate", strategy.generate, creator.Individual)
toolbox.register("update", strategy.update)

rows = np.repeat(np.arange(p+2).reshape(-1,1), p+2, axis=1)
columns = np.repeat(np.arange(p+2).reshape(1,-1), p+2, axis=0)
labels = list(map(lambda x,y : f"{x}->{y}", columns.ravel(), rows.ravel()))
labels = [label.replace(str(p+1),'inh') for label in labels]
def animate_weights(i, cma_strategy, ax, cmap=plt.cm.cool):
    ax.clear()
    start_weights = get_weights_from_genome(cma_strategy.stored_centroids[0]).ravel()
    end_weights = get_weights_from_genome(cma_strategy.stored_centroids[-1]).ravel()
    weights = get_weights_from_genome(cma_strategy.stored_centroids[i]).ravel()
    ax.plot(
        start_weights, '--',
        # color='orange'
        color="gray",
        alpha=0.5,
        label='start'
        )
    ax.plot(
        end_weights, '--',
        # color='orange'
        color="black",
        alpha=0.5,
        label='end'
        )
    ax.plot(
        weights, '*-',
        # color='orange'
        color=cmap(i/(n_gen-1))
        )

    # x_axis_vals = np.arange(weights.shape[0])
    # fixed_weights = x_axis_vals[columns.ravel()==3]
    # ax.plot(fixed_weights, np.ones_like(fixed_weights), 'ko')

    ax.set_xticks(np.arange(weights.shape[0]))
    ax.set_xticklabels(labels=labels, rotation=65)
    ax.set_ylabel('weight (a.u.)')
    ax.set_ylim(-.1, w_max_default+.1)
    ax.set_title(
        "Evolution of Synaptic Weights" +\
        f"\ncoherence:{args.coherence:.2f}, penalty:{penalty:.2f}, " +\
        f"runs:{n_runs}, restarts:{n_multiples}" +\
        f"\nTask: {task}"
        )
    ax.legend(loc='upper right')
    ax.grid(alpha=0.1, linestyle=':')


if __name__ == '__main__':
    if not path.exists(folder_prefix):
        mkdir(folder_prefix)
    if not path.exists(imagedir):
        mkdir(imagedir)
    if not path.exists(paramsdir):
        mkdir(paramsdir)
    # if not path.exists(checkpointsdir):
    #     mkdir(checkpointsdir)
    experiment_dict = dict(
        script=__file__,
        n_workers=n_workers,
        w_plus_initial=w_plus,
        w_minus_initial=w_minus,
        centroid_initial=centroid,
        input_args=dict(vars(args))
    )
    parameters_dict.update(experiment_dict)
    with open(paramsfile, 'w') as fp:
        json.dump(parameters_dict, fp)
    
    np.set_printoptions(precision=3, suppress=True)
    running_datetime = datetime.now()

    print("Confirm correct starting weights:")
    print(get_weights_from_genome(centroid))

    cluster = LocalCluster(n_workers=n_workers)
    client = Client(cluster)

    def dask_map(func, *seqs, **kwargs):
        results_future = client.map(func, *seqs, **kwargs)
        return client.gather(results_future)
    
    print("\nStarting EA...")
    start = time()
    toolbox.register("map", dask_map)
    pop, logbook = algorithms.eaGenerateUpdate(
        toolbox,
        ngen=n_gen,
        # stats=mstats,
        stats=stats,
        # halloffame=hof,
        verbose=True
    )

    end = time()
    print("Complete")
    print(f"Total Time Taken: {end-start:.2f} seconds")
    
    hof = strategy.halloffame
    print(
        "\nHall of Fame Weights:",
        *map(get_weights_from_genome, hof),
        sep='\n'
       )
    print(
        "\nFinal Population:",
        *map(get_weights_from_genome, pop),
        sep='\n'
    )

    fig, axes = plt.subplots(figsize=(12,6))
    func_animation = partial(animate_weights, cma_strategy=strategy, ax=axes)
    
    anim = animation.FuncAnimation(
        fig=fig,
        func=func_animation,
        frames=np.arange(n_gen+1)
        )
    if args.show_anim:
        plt.show()
    anim.save(path.join(
        imagedir, 'weights_animation.gif'
        ))
    
