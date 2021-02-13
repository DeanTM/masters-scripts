from evolution import *

from functools import partial

from dask.distributed import Client, LocalCluster
from deap import base, creator, tools, algorithms, cma

from time import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import animation

# from scipy.optimize import fmin

# W_opt = [[0.824, 1.017, 0.805, 1.   ],
#         [0.981, 2.081, 0.772, 1.   ],
#         [0.784, 0.958, 2.113, 1.   ],
#         [0.931, 0.92,  1.073, 1.   ]]

n_runs = 20
n_multiples = 2
coherence = 0.5
penalty = 0.5

n_gen = 100
n_hof = 20
lambda_EA = 12
sigma_EA = 1e-1

run_trial_func = run_trial_coherence_2afc
coherence = [-coherence, coherence]

w_plus = 2.1  # 1.
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

# moved to evolution.py
# class Genome(list):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.randomstate = None  #RandomState(MT19937(SeedSequence(seed)))


# # reuse random states
# # TODO: inherit this from evolution.py
# class CMAStrategyCustom(cma.Strategy):
#     def __init__(self, *args, store_centroids=False, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.randomstates = [RandomState() for i in range(self.lambda_)]
#         self.store_centroids = store_centroids
#         self.stored_centroids = None
#         if self.store_centroids:
#             self.stored_centroids = [self.centroid]
    
#     def generate(self, *args, **kwargs):
#         pop = super().generate(*args, **kwargs)
#         for i in range(self.lambda_):
#             pop[i].randomstate = self.randomstates[i]
#         return pop
    
#     def update(self, *args, **kwargs):
#         super().update(*args, **kwargs)
#         if self.store_centroids:
#             self.stored_centroids.append(self.centroid)
    
    


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
# strategy = cma.Strategy(
#     centroid=centroid,
#     lambda_=lambda_EA,
#     sigma=sigma_EA,
#     # weights='linear'  # does this slow down convergence?
# )
strategy = CMAStrategyCustom(
    centroid=centroid,
    lambda_=lambda_EA,
    sigma=sigma_EA,
    store_centroids=True,
    weights='linear'  # don't want to overemphasize chance
)

hof = tools.HallOfFame(n_hof)

toolbox = base.Toolbox()
toolbox.register("evaluate", fitness)
toolbox.register("generate", strategy.generate, creator.Individual)
toolbox.register("update", strategy.update)

rows = np.repeat(np.arange(p+2).reshape(-1,1), 4, axis=1)
columns = np.repeat(np.arange(p+2).reshape(1,-1), 4, axis=0)
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
    x_axis_vals = np.arange(weights.shape[0])
    fixed_weights = x_axis_vals[columns.ravel()==3]
    
    plt.plot(fixed_weights, np.ones_like(fixed_weights), 'ko')

    ax.set_xticks(np.arange(weights.shape[0]))
    ax.set_xticklabels(labels=labels, rotation=65)
    ax.set_ylabel('weight (a.u.)')
    ax.set_ylim(-.1, w_max_default+.1)
    ax.set_title(
        "Evolution of Synaptic Weights\n" +\
        f"coherence:{coherence[-1]:.2f},\tpenalty:{penalty:.2f}"
        )
    ax.legend(loc='upper right')
    ax.grid(alpha=0.1, linestyle=':')


if __name__ == '__main__':
    
    np.set_printoptions(precision=3, suppress=True)
    running_datetime = datetime.now()

    print("Confirm correct weights:")
    print(get_weights_from_genome(centroid))

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
    
    animation = animation.FuncAnimation(
        fig=fig,
        func=func_animation,
        frames=np.arange(n_gen+1)
        )
    
    plt.show()
    animation.save(f'weights_animation_{running_datetime}.gif')
