from evolution import * #Genome, get_params_from_genome, run_repeated_trial
# from simulation import run_trial_coherence_2afc
# from functions import get_weights, get_w_minus
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
import argparse

from dask.distributed import Client

from datetime import datetime
from time import time
from os import path, listdir, mkdir

parser = argparse.ArgumentParser()
parser.add_argument('--maxwplus', type=float, default=1.)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--numworkers', type=int, default=1)
parser.add_argument('-p', type=int, default=2)
parser.add_argument('-m', '--mingen', type=int, default=30)
parser.add_argument('-n', '--samples', type=int, default=10)
parser.add_argument('-w', '--weights', action='store_true')
parser.add_argument('-s', '--scaleup', action='store_true')
parser.add_argument('--savesamples', action='store_true')
parser.add_argument('--noeval', action='store_true')
args = parser.parse_args()

script_running_datetime = str(datetime.now()).replace(' ', '_')
print(f"Starting {__file__} at {script_running_datetime}")
folder_suffix = '_'.join([__file__[:-3], script_running_datetime])
folder_prefix = path.join(path.join('experiments', folder_suffix))
imagedir = path.join(folder_prefix, 'images_and_animations')
samplesdir = path.join(folder_prefix, 'samples')

# selection criteria
min_generations = args.mingen
maxwplus = args.maxwplus
p_choice = args.p
num_samples = args.samples
experiment_type = 'run_evolution' if not args.weights else 'find_optimal_weights'
multiplier = 10. if experiment_type == 'find_optimal_weights' and args.scaleup else 1.
evaluate_samples = not args.noeval
n_multiples = 1
n_runs = 100
penalty = 0.5
w_plus = 1.
n_workers = args.numworkers

def get_checkpoint_and_params(experiment_fname):
    lr_evolution_base = path.join(path.curdir, 'experiments')
    lr_evolution_prefix = path.join(lr_evolution_base, experiment_fname)
    lr_evolution_checkpoints = path.join(lr_evolution_prefix, 'checkpoints')
    lr_evolution_parameters = path.join(lr_evolution_prefix, 'parameters')
    if not path.exists(lr_evolution_checkpoints) or not path.exists(lr_evolution_parameters):
        return None
    lr_checkpoints_fnames = sorted(
        [path.join(lr_evolution_checkpoints, x) for x in listdir(lr_evolution_checkpoints)],
        key=path.getmtime)
    if len(lr_checkpoints_fnames) == 0:
        print(
            "No checkpoints found in folder for experiment: ",
            experiment_fname)
        return None

    with open(lr_checkpoints_fnames[-1], 'r') as f:
        checkpoint = json.load(f)
    with open(path.join(
        lr_evolution_parameters,
        listdir(lr_evolution_parameters)[0]), 'r') as f:
        params = json.load(f)
    return checkpoint, params

get_rewards = partial(
    get_reward_from_results,
    penalty=penalty, n_runs=n_runs
)
W_initial = get_weights(
    w_plus=w_plus,
    w_minus=get_w_minus(w_plus=w_plus)
    )

def plasticity_fitness(
    plasticity_params,
    coherence,
    n_runs=n_runs, n_multiples=n_multiples,
    W_initial=W_initial,
    randomstate=None,
):  
    trial_func = partial(
        run_trial_coherence_2afc,
        total_time=2*runtime,
        use_phi_fitted=True,
        coherence=coherence
    )
    all_results = [run_repeated_trial(
        W_initial=W_initial,
        plasticity_params=plasticity_params,
        trial_func=trial_func, n_runs=n_runs,
        verbose=False,
        randomstate=randomstate,
        nan_verbose=True
        )
        for i in range(n_multiples)]
    fitness = 0.
    for j, results_dict in enumerate(all_results):
        fitness += get_rewards(results_dict)
    return fitness / n_multiples

def fitness_learningrule(
    genome, coherence, n_runs=n_runs, n_multiples=n_multiples
    ):
    plasticity_params = get_params_from_genome(np.array(genome))
    randomstate = genome.randomstate
    fitness = plasticity_fitness(
        plasticity_params=plasticity_params,
        coherence=coherence,
        n_runs=n_runs,
        n_multiples=n_multiples,
        randomstate=randomstate,
        )
    return fitness

def weight_fitness(
    W, coherence,
    n_runs=n_runs, n_multiples=n_multiples,
    randomstate=None
):
    trial_func = partial(
        run_trial_coherence_2afc,
        total_time=2*runtime,
        use_phi_fitted=True,
        coherence=coherence
    )
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

@jit(nopython=True)
def bound_weights(W):
    return w_max_default * sigmoid(W)


def get_weights_from_genome(genome):
    genome_reshaped = np.array(genome).reshape(p+2,p+1)
    bound_genome = bound_weights(genome_reshaped)
    return np.hstack([bound_genome, W_initial[:,-1].reshape(-1,1)])


def fitness_weights(
    genome, coherence, n_runs=n_runs, n_multiples=n_multiples
):
    W = get_weights_from_genome(genome)
    randomstate = genome.randomstate
    fitness = weight_fitness(
        W=W,
        coherence=coherence,
        randomstate=randomstate,
        n_runs=n_runs, n_multiples=n_multiples
        )
    return fitness

evaluate = fitness_learningrule if experiment_type == 'run_evolution' else fitness_weights


if __name__ == "__main__":
    if not path.exists(folder_prefix):
        mkdir(folder_prefix)
    if not path.exists(imagedir):
        mkdir(imagedir)

    client = Client(n_workers=n_workers)
    # choosing experiments
    experiments = [x for x in listdir('experiments') if x.startswith(experiment_type)]
    checks_and_params = [get_checkpoint_and_params(x) for x in experiments]
    checks_and_params = [x for x in checks_and_params if x is not None]
    checks_and_params = sorted(
        checks_and_params,
        key=lambda x: x[1]['input_args']['coherence'],
        reverse=True
        )
    # extracting features
    checks_and_params_filtered = []
    coherences = []
    first_hof_entries = []
    fitness_indices = []
    best_fitnesses = []
    best_fitness_std = []
    for checkpoint, params in checks_and_params:
        num_generations = np.array(checkpoint['fitness_avg']).shape[0]
        p = params['p']
        w_plus = params['input_args'].get('w_plus', 1.0)  # default untrained
        start_trained = params['input_args']['start_trained']
        if num_generations >= min_generations and p == p_choice \
            and w_plus <= maxwplus and not start_trained:
            coherences.append(params['input_args']['coherence'])
            checks_and_params_filtered.append( (checkpoint, params) )
            first_hof_member = Genome(checkpoint['halloffame'][0])
            first_hof_entries.append(first_hof_member)
            fitness_avg = np.array(checkpoint['fitness_avg']) * multiplier
            fitness_std = np.array(checkpoint['fitness_std']) * multiplier
            fitness_index = np.argmax(fitness_avg-fitness_std)  # get index of at -1std
            fitness_indices.append(fitness_index)
            best_fitnesses.append(fitness_avg[fitness_index])
            best_fitness_std.append(fitness_std[fitness_index])
    
    plt.figure(figsize=(8, 5))
    for c, hof_member in zip(coherences, first_hof_entries):
        plt.plot(hof_member, label=c)
    plt.legend(title='coherence')
    if experiment_type == 'run_evolution':
        savename = 'best_genomes_learning_rules.png'
        title = "Genomes of Best Learning Rules"
    else:
        savename = 'best_genomes_weights.png'
        title = "Genomes of Best Weights"
    savename = path.join(imagedir, savename)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savename)
    # plt.show()

    plt.figure(figsize=(8, 5))
    best_fitnesses = np.array(best_fitnesses)
    best_fitness_std = np.array(best_fitness_std)
    plt.errorbar(
        coherences,
        best_fitnesses,
        yerr=best_fitness_std,
        color='gray',
        label='Population Average')
    fitness_samples = np.zeros(shape=(len(coherences), num_samples))
    if evaluate_samples:
        print(f"Starting sampling using function {evaluate}...")
        first_hof_entries = [x.update_randomstate() for x in first_hof_entries]
        states = [x.randomstate.get_state()[1] for x in first_hof_entries]
        print("Random States:")
        print(np.vstack(states))
        start = time()
        for n in tqdm(range(num_samples)):
            first_hof_entries = [x.update_randomstate() for x in first_hof_entries]
            results = client.map(evaluate, first_hof_entries, coherences)
            results = client.gather(results)
            fitness_samples[:, n] = np.array(results)
        # for i, c in enumerate(tqdm(coherences)):
        #     # parallelise sampling across samples
        #     hof_candidate = first_hof_entries[i].update_randomstate(seed=hash(i)+hash(c)%2**32)
        #     results = np.zeros(num_samples)
        #     results = [
        #         client.submit(evaluate, hof_candidate, coherence=c)
        #         # evaluate(first_hof_entries[i], coherence=c)
        #         for sample in range(num_samples)
        #         ]
        #     results = client.gather(results)
        #     fitness_samples[i, :] = np.array(results)
        end = time()
        print(f"Sampling complete in {end-start:.2f}s")
        print("Fitness samples:")
        print(fitness_samples)
        if args.savesamples:
            if not path.exists(samplesdir):
                mkdir(samplesdir)
            coherences_array = np.array(coherences)
            first_hof_array = np.array(list(x) for x in first_hof_entries)
            np.save(path.join(samplesdir, 'samples.npy'), fitness_samples)
            np.save(path.join(samplesdir, 'coherences.npy'), coherences_array)
            np.save(path.join(samplesdir, 'genomes.npy'), first_hof_array)

        for i, c in enumerate(coherences):
            x_vals = np.full_like(fitness_samples[i,:], c)
            label = None
            if i == 0:
                label = 'Sample Performance'
            plt.plot(
                x_vals,
                fitness_samples[i,:], 'k.',
                label=label,
                markersize=4.,
                alpha=args.alpha
                )
    plt.legend()
    plt.xticks(coherences, rotation=45)
    plt.xlabel('coherence')
    plt.ylabel('fitness (number of correct trials out of 100)')
    plt.title("Performances of Best Individuals and Generations")
    plt.grid(ls=':', alpha=0.2)
    plt.tight_layout()
    if experiment_type == 'run_evolution':
        savename = 'fitness_samples_learning_rule.png'
    else:
        savename = 'fitness_samples_weights.png'
    savename = path.join(imagedir, savename)
    plt.savefig(savename)
    plt.show()
    