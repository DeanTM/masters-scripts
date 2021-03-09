# TODO: this test file is not up to date
# but not needed, as run_evolution.py works
from evolution import *

from functools import partial
from time import time

# import matplotlib.pyplot as plt
# from datetime import datetime

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from deap import cma

from dask import delayed, compute
import dask.multiprocessing
import dask.bag as db

import argparse

parser = argparse.ArgumentParser(
    description = 'testing parallelisation of EA'
)
parallel_group = parser.add_mutually_exclusive_group()
parallel_group.add_argument(
    '--pm',
    action='store_true',
    help='whether to use custom parallel map function. ' +\
         'Seems faster than --pr'
)
parallel_group.add_argument(
    '--db',
    action='store_true',
    help='whether to use dask.bag.map'
)
parallel_group.add_argument(
    '--pr',
    action='store_true',
    help='whether to use parallel restarts'
)
parser.add_argument(
    '-n', '--n_gen',
    type=int, default=100,
    help='number of generations to run the EA'
)
parser.add_argument(
    '-r', '--n_runs',
    type=int, default=10,
    help='number of runs of the trial'
)
parser.add_argument(
    '-m', '--n_multiples',
    type=int, default=10,
    help='number of repeats of the trial'
)
parser.add_argument(
    '--sigma',
    type=float, default=0.0005,
    help='initial sigma for the EA'
)
parser.add_argument(
    '-v', '--verbose',
    action='store_true',
    help='whether to print progress'
)
parser.add_argument(
    '--scheduler',
    type=str, default='processes',
    help="Scheduler for dask. " +\
         "Options are ['threads', 'processes', 'synchronous']"
)

args = parser.parse_args()

assert args.scheduler in ['threads', 'processes', 'synchronous']
dask.config.set(scheduler=args.scheduler)


def map_dask(f, *iters):
    """
    Following the solution here: 
    https://stackoverflow.com/questions/42051318/programming-deap-with-scoop
    
    Use with `toolbox.register("map", map_dask)`
    """
    f_delayed = delayed(f)
    return compute(*[f_delayed(*args) for args in zip(*iters)])

def map_dask_bag(f, seq):
    return db.from_sequence(seq).map(f).compute()

def get_fitness_dask(
    W_initial, plasticity_params,
    trial_func,
    n_runs,
    n_multiples,
    verbose=False
):
    fitness = 0.0
    
    # very crude parallelisation
    run_repeated_trial_delayed = delayed(run_repeated_trial)
    computations = [run_repeated_trial_delayed(
            W_initial=W_initial,
            plasticity_params=plasticity_params,
            trial_func=trial_func,
            n_runs=n_runs,
            verbose=verbose)
        for i in range(n_multiples)]
    
    results_dicts = compute(computations)[0]
    for results_dict in results_dicts:
        for rew_array in results_dict['reward']:
            if not np.any(np.isnan(rew_array)):
                fitness += np.sum(rew_array)
    return fitness



#### THIS DEFINES THE EA TO RUN IN PARALLEL ####

trial_func = partial(
    run_trial_coherence_2afc,
    total_time=runtime,
    coherence=0.068,
    use_phi_fitted=True
)

W_initial = get_weights()
if args.pr:
    def fitness_EA(genome):
        plasticity_params = get_params_from_genome(
            np.array(genome)
        )
        return get_fitness_dask(
            W_initial=W_initial,
            plasticity_params=plasticity_params,
            n_runs=args.n_runs,
            n_multiples=args.n_multiples,
            trial_func=trial_func
        ),
else:
    def fitness_EA(genome):
        plasticity_params = get_params_from_genome(
            np.array(genome)
        )
        return get_fitness(
            W_initial=W_initial,
            plasticity_params=plasticity_params,
            n_runs=args.n_runs,
            n_multiples=args.n_multiples,
            trial_func=trial_func
        ),
###############################################

creator.create("FitnessMax", base.Fitness, weights=(1.,))
creator.create("Individual", list, fitness=creator.FitnessMax)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

strategy = cma.Strategy(
    centroid=nolearn_genome,
    sigma=args.sigma
)

hof = tools.HallOfFame(10)

toolbox = base.Toolbox()
toolbox.register("evaluate", fitness_EA)
toolbox.register("generate", strategy.generate, creator.Individual)
toolbox.register("update", strategy.update)

if args.pm:
    toolbox.register("map", map_dask)
elif args.db:
    toolbox.register("map", map_dask_bag)
    
if __name__ == '__main__':
    start = time()
    pop, logbook = algorithms.eaGenerateUpdate(
        toolbox,
        ngen=args.n_gen,
        stats=stats,
        halloffame=hof,
        verbose=args.verbose
    )
    end = time()
#     print(logbook)
    print("\nEvolutionary algorithm complete with args:")
    print(*map(lambda x: f"{x[0]}:{x[1]}", vars(args).items()))
    print(f"Total time taken: {end-start:.2f}s")