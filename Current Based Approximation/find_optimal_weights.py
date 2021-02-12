from evolution import *

from functools import partial

from dask.distributed import Client, LocalCluster
from deap import base, creator, tools, algorithms, cma

from time import time

from scipy.optimize import fmin


n_runs = 20
n_multiples = 1
coherence = 0.8
penalty = 0.5

n_gen = 100
n_hof = 20
lambda_EA = 12
sigma_EA = 1e-3

run_trial_func = run_trial_coherence_2afc
coherence = [-coherence, coherence]

w_plus = 1.
w_minus = get_w_minus(w_plus)
W_initial = get_weights(
    w_plus=w_plus,
    w_minus=w_minus
)

trial_func = partial(
    run_trial_func,
    total_time=2*runtime,
    coherence=coherence,
    use_phi_fitted=True,
    plasticity=False,
)

get_rewards = partial(
    get_reward_from_results,
    penalty=penalty, n_runs=n_runs
)


@jit(nopython=True)
def bound_weights(W):
    return w_max_default * sigmoid(W)

# invert bound_weights at w_plus, w_minus
# this reruns in each process, so I fix them to avoid this
# w_plus_unbound = fmin(
#     lambda w: (bound_weights(w) - w_plus)**2,
#     x0 = w_plus,
#     xtol = 1e-12,
#     ftol=1e-12
#     )
# w_minus_unbound = fmin(
#     lambda w: (bound_weights(w) - w_minus)**2,
#     x0 = w_plus,
#     xtol = 1e-12,
#     ftol=1e-12
#     )
# unit_unbound = fmin(
#     lambda w: (bound_weights(w) - 1.)**2,
#     x0 = w_plus,
#     xtol = 1e-12,
#     ftol=1e-12
#     )
# w_plus_unbound = w_minus_unbound = unit_unbound = -0.18232156
unit_unbound = -0.18232156  # bound_weights^-1(1)
w_plus_unbound = 3.04453857  # bound_weights^-1(2.1)
w_minus_unbound = -0.40969482  # bound_weights^-1(get_w_minus(2.1))

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
        nan_verbose=False,
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


class Genome(list):
    def __init__(self, *args, seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.randomstate = RandomState(MT19937(SeedSequence(seed)))

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
mstats = tools.MultiStatistics(fitnesses=stats, w_max=stats_weights_max, w_min=stats_weights_min)
mstats.register('max', np.max)
mstats.register('min', np.min)

# centroid = list(W_initial.ravel())
# centroid = [-0.18232156 for i in range(W_initial.ravel().shape[0])]
centroid = np.full_like(W_initial[:, :-1], unit_unbound)
centroid[W_initial[:, :-1] == w_plus] = w_plus_unbound
centroid[W_initial[:, :-1] == w_minus] = w_minus_unbound
centroid = [x for x in centroid.ravel()]  # don't have plastic weights change
strategy = cma.Strategy(
    centroid=centroid,
    lambda_=lambda_EA,
    sigma=sigma_EA,
    # weights='linear'  # does this slow down convergence?
)

hof = tools.HallOfFame(n_hof)

toolbox = base.Toolbox()
toolbox.register("evaluate", fitness)
toolbox.register("generate", strategy.generate, creator.Individual)
toolbox.register("update", strategy.update)

if __name__ == '__main__':
    print("Confirm correct weights:")
    print(get_params_from_genome(centroid))

    cluster = LocalCluster(n_workers=lambda_EA)
    client = Client(cluster)

    def dask_map(func, *seqs, **kwargs):
        results_future = client.map(func, *seqs, **kwargs)
        return client.gather(results_future)
    
    start = time()

    toolbox.register("map", dask_map)
    pop, logbook = algorithms.eaGenerateUpdate(
        toolbox,
        ngen=n_gen,
        stats=mstats,
        halloffame=hof,
        verbose=True
    )

    end = time()
    print("Complete")
    print(f"Total Time Taken: {end-start:.2f} seconds")
    
    np.set_printoptions(precision=3)
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
