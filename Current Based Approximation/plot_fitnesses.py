from os import path, listdir
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights', action='store_true')
args = parser.parse_args()

def get_checkpoint_and_params(experiment_fname):
    print(experiment_fname)
    lr_evolution_base = '/home/dean/Projects/plasticity-code/Current Based Approximation/experiments/'
    lr_evolution_prefix = lr_evolution_base + experiment_fname
    lr_evolution_checkpoints = path.join(lr_evolution_prefix, 'checkpoints')
    lr_evolution_parameters = path.join(lr_evolution_prefix, 'parameters')
    lr_checkpoints_fnames = sorted(
        [path.join(lr_evolution_checkpoints, x) for x in listdir(lr_evolution_checkpoints)],
        key=path.getmtime)
    if len(lr_checkpoints_fnames) == 0:
        print(
            "No checkpoints found in folder for experiment: ",
            experiment_fname)
        return

    with open(lr_checkpoints_fnames[-1], 'r') as f:
        checkpoint = json.load(f)
    with open(path.join(
        lr_evolution_parameters,
        listdir(lr_evolution_parameters)[0]), 'r') as f:
        params = json.load(f)
    return checkpoint, params

# def plot_fitness_curve(experiment_fname, ax=None):
#     checkpoint, params = get_checkpoint_and_params(experiment_fname)
def plot_fitness_curve(checkpoint, params, ax=None):
    coherence = params['input_args']['coherence']

    fitness_avg = np.array(checkpoint['fitness_avg'])
    fitness_std = np.array(checkpoint['fitness_std'])

    generations = np.arange(fitness_avg.shape[0])
    if ax is None:
        ax = plt.gca()
    l1 = ax.plot(
        generations,
        fitness_avg,
        label=f'{coherence:.2f}'
    )
    l2 = ax.fill_between(
        generations,
        fitness_avg-fitness_std,
        fitness_avg+fitness_std,
        alpha=0.3,
        color=l1[0].get_color(),
    )
    return l1, l2, ax


if __name__ == '__main__':
    if args.weights:
        experiment_type = 'find_optimal_weights'
    else:
        experiment_type = 'run_evolution'
    plt.figure(figsize=(8, 5))
    experiments = [x for x in listdir('experiments') if x.startswith(experiment_type)]
    checks_and_params = [get_checkpoint_and_params(x) for x in experiments]
    checks_and_params = [x for x in checks_and_params if x is not None]
    checks_and_params = sorted(
        checks_and_params,
        key=lambda x: x[1]['input_args']['coherence']
        )

    for checkpoint, params in checks_and_params:
        try:
            plot_fitness_curve(checkpoint, params)
        except FileNotFoundError as e:
            print("No checkpoints found for experiment: ", experiment)
    plt.grid(ls=':', alpha=.5)
    plt.legend(title='task coherence', loc='lower right')
    title = "Performance of Evolved Synaptic Weights" if args.weights else "Performance of Evolved Learning Rules"
    plt.title(title)
    plt.hlines(10. if args.weights else 100., 0., 50., ls='--', color='k')
    plt.ylabel("Fitness")
    plt.xlabel("Generation")
    fname = 'weights_fitnesses.png' if args.weights else 'learning_rules_fitnesses.png'
    plt.savefig('images_and_animations/learning_rules_fitnesses.png')
    plt.show()