from os import path, listdir
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights', action='store_true')
parser.add_argument('--maxwplus', type=float, default=1.5)
parser.add_argument('--alpha', type=float, default=0.3)
parser.add_argument('--varalpha', action='store_true')
parser.add_argument('--noweightfilter', action='store_true')
parser.add_argument('-m', '--mingen', type=int, default=30)
parser.add_argument('-t', '--truncate', type=int, default=40)
parser.add_argument('-s', '--scaleup', action='store_true')
parser.add_argument('-l', '--labelsigma', action='store_true')
parser.add_argument('--stdwidth', type=float, default=1.)
parser.add_argument('-p', type=int, default=2)
parser.add_argument('--noshow', action='store_true')
args = parser.parse_args()

#TODO: put helper functions in a separate file
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

    for i, json_fname in enumerate(reversed(lr_checkpoints_fnames)):
        try:
            with open(json_fname, 'r') as f:
                checkpoint = json.load(f)
                break
        except json.JSONDecodeError:
            print(f"JSON decode error on: {experiment_fname} - {json_fname}")
            if i == len(lr_checkpoints_fnames) - 1:
                return None
    # with open(lr_checkpoints_fnames[-1], 'r') as f:
            # return None
    with open(path.join(
        lr_evolution_parameters,
        listdir(lr_evolution_parameters)[0]), 'r') as f:
        params = json.load(f)
    return checkpoint, params

# def plot_fitness_curve(experiment_fname, ax=None):
#     checkpoint, params = get_checkpoint_and_params(experiment_fname)
def plot_fitness_curve(
    checkpoint, params,
    ax=None, multiplier=1.,
    label_sigma=False,
    alpha_full=0.3,
    variable_alpha=True,
    stdwidth=1.,
    colour=None
    ):
    coherence = params['input_args']['coherence']

    fitness_avg = np.array(checkpoint['fitness_avg']) * multiplier
    fitness_std = np.array(checkpoint['fitness_std']) * multiplier

    generations = np.arange(fitness_avg.shape[0])
    if ax is None:
        ax = plt.gca()
    line_label = f'{coherence:.2f}'
    if label_sigma:
        sigma = params['input_args']['sigma']
        line_label += f' | {sigma:.2f}'
    l1 = ax.plot(
        generations,
        fitness_avg,
        label=line_label,
        color=colour
    )
    if not variable_alpha:
        l2 = ax.fill_between(
            generations,
            fitness_avg-stdwidth*fitness_std,
            fitness_avg+stdwidth*fitness_std,
            alpha=alpha_full,
            color=l1[0].get_color(),
        )
    else:
        for g in range(generations.shape[0]):
            alpha_here = alpha_full * np.sqrt(10/(1+np.mean(fitness_std[g:g+2])))
            if args.weights and not args.scaleup:
                alpha_here = alpha_here / np.sqrt(10)
            l2 = ax.fill_between(
                generations[g:g+2],
                fitness_avg[g:g+2]-stdwidth*fitness_std[g:g+2],
                fitness_avg[g:g+2]+stdwidth*fitness_std[g:g+2],
                alpha=alpha_here,
                color=l1[0].get_color(),
                edgecolor="none"
            )
    return ax


if __name__ == '__main__':
    min_generations = args.mingen  # filter out short experiments
    truncate_plot = args.truncate  # truncate long experiments
    multiplier = 1.
    if args.weights:
        experiment_type = 'find_optimal_weights'
        if args.scaleup:
            multiplier = 10.
    else:
        experiment_type = 'run_evolution'
    plt.figure(figsize=(8, 5))
    experiments = [x for x in listdir('experiments') if x.startswith(experiment_type)]
    checks_and_params = [get_checkpoint_and_params(x) for x in experiments]
    checks_and_params = [x for x in checks_and_params if x is not None]
    checks_and_params = sorted(
        checks_and_params,
        key=lambda x: x[1]['input_args']['coherence'],
        reverse=True
        )

    # checks_and_params_filtered = []
    best_per_coherence = {}
    best_scores_per_coherence = {}
    best_mean_std_per_coherence = {}
    for checkpoint, params in checks_and_params:
        num_generations = np.array(checkpoint['fitness_avg']).shape[0]
        p = params['p']
        w_plus = params['input_args'].get('w_plus', 1.0)  # default untrained
        start_trained = params['input_args']['start_trained']
        if num_generations >= min_generations and p == args.p \
            and (w_plus <= args.maxwplus or args.noweightfilter) and not start_trained:
            # first_hof_member = checkpoint['halloffame'][0]
            # first_hof_entries.append(first_hof_member)
            fitness_avg = np.array(checkpoint['fitness_avg']) * multiplier
            fitness_std = np.array(checkpoint['fitness_std']) * multiplier
            # fitness_index = np.argmax(fitness_avg-fitness_std)
            best_fitness = np.max(fitness_avg-fitness_std)
            best_fitness_idx = np.argmax(fitness_avg-fitness_std)
            coherence = params['input_args']['coherence']
            if coherence not in best_scores_per_coherence:
                best_per_coherence[coherence] = (checkpoint, params)
                best_scores_per_coherence[coherence] = best_fitness
                best_mean_std_per_coherence[coherence] = [[
                    fitness_avg[best_fitness_idx],
                    fitness_std[best_fitness_idx]]]
            elif best_fitness > best_scores_per_coherence[coherence]:
                best_per_coherence[coherence] = (checkpoint, params)
                best_scores_per_coherence[coherence] = best_fitness
                best_mean_std_per_coherence[coherence] = [[
                    fitness_avg[best_fitness_idx],
                    fitness_std[best_fitness_idx]]]
            # fitness_indices.append(fitness_index)
            # best_fitnesses.append(fitness_avg[fitness_index])
            # best_fitness_std.append(fitness_std[fitness_index])

            # checks_and_params_filtered.append( (checkpoint, params) )
    # save for convenience
    coherences = np.array(sorted(best_mean_std_per_coherence.keys()))
    best_means_stds_array = np.empty((len(coherences), 2))
    for i, c in enumerate(coherences):
        best_means_stds_array[i, :] = np.array(best_mean_std_per_coherence[c])
    np.save(experiment_type+'_crosssection_meanstd.npy', best_means_stds_array)
    np.save(experiment_type+'_crosssection_coherence.npy', coherences)

    checks_and_params_filtered = list(best_per_coherence.values())
    for i, (checkpoint, params) in enumerate(checks_and_params_filtered):
        # colour = cmap(i/len(checks_and_params_filtered))
        plot_fitness_curve(
            checkpoint, params,
            multiplier=multiplier,
            label_sigma=args.labelsigma,
            alpha_full=args.alpha,
            variable_alpha=args.varalpha,
            stdwidth=args.stdwidth,
            # match up colours for different coherences
            colour=colours[(i+4) % len(colours) if args.weights else i %len(colours)]
            )

    plt.grid(ls=':', alpha=.5)
    legend_title = title='coherence'
    if args.labelsigma:
        legend_title += ' | sigma'
    plt.legend(title=legend_title, loc='best', ncol=2)
    plt.xlim(0, truncate_plot)
    title = "Performance of Evolved Synaptic Weights" if args.weights else "Performance of Evolved Learning Rules"
    if args.weights and args.scaleup:
        title += '\nscaled to fitness out of 100'
    plt.title(title)
    # plt.hlines(10. if args.weights else 100., 0., 50., ls='--', color='k')
    plt.axhline(10. if args.weights and not args.scaleup else 100., ls='--', color='k')
    plt.ylabel("Fitness")
    plt.xlabel("Generation")
    fname = 'weights_fitnesses.png' if args.weights else 'learning_rules_fitnesses.png'
    plt.savefig(path.join('images_and_animations', fname))
    if not args.noshow:
        plt.show()
