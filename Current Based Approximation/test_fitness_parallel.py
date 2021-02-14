from evolution import *

from functools import partial
import time
from datetime import datetime
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dask.distributed import Client, LocalCluster
from os import path, listdir

import argparse
# from dask.diagnostics import ProgressBar

# from numpy.random import SeedSequence  # do better randomness handling

parser = argparse.ArgumentParser(
    description='Test variance and means'
    )
parser.add_argument(
    '--plotonly', action='store_true',
    help='whether to only recreate last plot'
    )

args = parser.parse_args()


def get_reward_from_results(results_dict):
    reward = 0.
    for rew_array in results_dict['reward']:
            if not np.any(np.isnan(rew_array)):
                reward += np.sum(rew_array)
            else:
                print("There were NaNs")
    return reward * defaultdt  # normalise reward for different timestep sizes


def get_fitness_dask(
    W_initial, 
    plasticity_params,
    trial_func,
    n_runs,
    n_multiples,
    client,
    seeds=None
):
    if seeds is None:
        seeds = np.random.random_integers(n_multiples*100, size=n_multiples)
    result_dicts_futures = [
        client.submit(
            run_repeated_trial,
            W_initial=W_initial,
            plasticity_params=plasticity_params,
            trial_func=trial_func,
            n_runs=n_runs,
            verbose=False,
            seed=seeds[m],  # trial func kwarg, each repeated task should be diff.
        )
        for m in range(n_multiples)]
    all_rewards_futures = client.map(
        get_reward_from_results, result_dicts_futures
        )

    fitness = client.submit(np.mean, all_rewards_futures)
    return fitness


def estimate_fitness_stats(
    W_initial, 
    plasticity_params,
    trial_func,
    n_runs,
    n_multiples,
    n_samples,
    client
):
    samples = [get_fitness_dask(
            W_initial=W_initial,
            plasticity_params=nolearn_parameters,
            trial_func=trial_func,
            n_runs=n_runs,
            n_multiples=n_multiples,
            client=client,
            seeds=np.random.random_integers(
                n_multiples*100, size=n_multiples
                )
        )
        for n in range(n_samples)]
    # samples, max_rates = zip(*[
    #     client.submit(
    #         get_fitness_dask,
    #         W_initial=W_initial,
    #         plasticity_params=nolearn_parameters,
    #         trial_func=trial_func,
    #         n_runs=n_runs,
    #         n_multiples=n_multiples,
    #         client=client,
    #         seeds=np.random.random_integers(
    #             n_multiples*100, size=n_multiples
    #             )
    #     )
    #     for n in range(n_samples)])
    variance = client.submit(np.var, samples)
    mean = client.submit(np.mean, samples)
    return mean, variance, samples


def main(mode='simulate'):
    datadir = 'fitness_tests_data'

    w_plus = 1.1
    W_initial = get_weights(
        w_plus=w_plus,
        w_minus=get_w_minus(w_plus=w_plus)
        )

    n_samples = 32  # enough to estimate stats?
    n_runs = 1  # no learning for now
    # coherence_vals = np.linspace(0.01, 0.99, 20)
    coherence_vals = np.logspace(np.log10(.01), np.log10(.99), 20)
    n_multiples_values = np.arange(1, 10, dtype=int)
        

    if mode == 'simulate':
        cluster = LocalCluster(n_workers=16)
        client = Client(cluster)

        datetime_suffix = datetime.now()
        variance_matrix = np.empty(
            (coherence_vals.shape[0], n_multiples_values.shape[0])
            )
        mean_matrix = np.empty(
            (coherence_vals.shape[0], n_multiples_values.shape[0])
            )

        for i, coherence in enumerate(coherence_vals):
            print(
                "Estimating for coherence value "
                f"number {i}/{coherence_vals.shape[0]}: {coherence:.3f}"
                )
            trial_func = partial(
                run_trial_coherence_2afc,
                total_time=2.*runtime,
                coherence=coherence,
                use_phi_fitted=True
            )

            for j, n_multiples in enumerate(n_multiples_values):
                start = time.time()
                mean, variance, samples = estimate_fitness_stats(
                    W_initial=W_initial,
                    plasticity_params=nolearn_parameters,
                    trial_func=trial_func,
                    n_runs=n_runs,
                    n_multiples=n_multiples,
                    n_samples=n_samples,
                    client=client
                )
                
                mean_res, var_res = client.gather([mean, variance])
                end = time.time()

                variance_matrix[i, j] = var_res
                mean_matrix[i, j] = mean_res
                print(
                    f"coherence {coherence:.3f}, n_multiples {n_multiples}: "
                    f"mean {mean_res:.3f}, var {var_res:.3f}, "
                    f"time {end-start}s"
                    )
        
        
        np.save(
            path.join(
                datadir,
                f'fitness-tests-means-{datetime_suffix}.npy'
            ), mean_matrix)
        np.save(
            path.join(
                datadir,
                f'fitness-tests-vars-{datetime_suffix}.npy'
            ), variance_matrix)
    
    elif mode=='plot_only':
        f_name_means = sorted([x for x in listdir(datadir) if 'means' in x])[-1]
        f_name_vars = sorted([x for x in listdir(datadir) if 'vars' in x])[-1]
        datetime_suffix = f_name_means.split('means-')[-1][:-4]
        with open(path.join(datadir,f_name_means), 'rb') as f:
            mean_matrix = np.load(f)
        with open(path.join(datadir,f_name_vars), 'rb') as f:
            variance_matrix = np.load(f)
    
    fig, axes = plt.subplots(1, 2, sharey=True)
    # img1 = axes[0].imshow(
    #     variance_matrix,
    #     extent=[n_multiples_values[0], n_multiples_values[-1],
    #         coherence_vals[0], coherence_vals[-1]],
    #     origin='lower'
    #     )
    X,Y = np.meshgrid(n_multiples_values, coherence_vals)
    
    img1 = axes[0].pcolormesh(X,Y,variance_matrix)
    axes[0].set_ylabel('coherence')
    axes[0].set_xlabel('number of restarts')
    axes[0].set_title('Variance in Performance')
    
    divider1 = make_axes_locatable(axes[0])
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img1, cax=cax1)
    
    # img2 = axes[1].imshow(
    #     mean_matrix,
    #     extent=[n_multiples_values[0], n_multiples_values[-1],
    #         coherence_vals[0], coherence_vals[-1]],
    #     origin='lower'
    #     )
    img2 = axes[1].pcolormesh(X,Y,mean_matrix)
    # axes[1].set_ylabel('coherence')
    axes[1].set_xlabel('number of restarts')
    axes[1].set_title('Means of Performance')
    divider2 = make_axes_locatable(axes[1])
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img2, cax=cax2)

    # plt.tight_layout()
    plt.savefig(f'fitness_variances_means_{datetime_suffix}.png')

    return None



if __name__ == '__main__':
    mode = 'plot_only' if args.plotonly else 'simulate'
    main(mode)    
