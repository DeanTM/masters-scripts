import numpy as np
from scipy.stats import ks_2samp, entropy
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
# from os import listdir, path

from evolution import *
from tqdm import tqdm
from dask.distributed import Client


#fixed samples
# prefix = 'plot_top_candidates_2021-02-28_19:12:06.488464'
num_samples_kstest = 40
samples_fname = 'experiments/plot_top_candidates_2021-02-28_19:12:06.488464/samples/samples.npy'
coherences_fname = 'experiments/plot_top_candidates_2021-02-28_19:12:06.488464/samples/coherences.npy'
genomes_fname = 'experiments/plot_top_candidates_2021-02-28_19:12:06.488464/samples/genomes.npy'
n_workers = 11

n_multiples = 1
n_runs = 100
penalty = 0.5
w_plus = 1.

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
    for results_dict in all_results:
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

evaluate = fitness_learningrule


if __name__ == '__main__':
    client = Client(n_workers=n_workers)

    fitness_samples = np.load(samples_fname)
    coherences = np.load(coherences_fname)
    genomes = np.load(genomes_fname)
    entropies = np.zeros(fitness_samples.shape[0])
    for i in range(fitness_samples.shape[0]):
        sample_1 = fitness_samples[i, :]
        hist_estimate, bins = np.histogram(sample_1, bins=20, range=(-100, 100), density=True)
        entropies[i] = entropy(hist_estimate)
    
    print(entropies)
    plt.figure(figsize=(8,3))
    plt.plot(
        coherences,
        entropies,
        'k*-'
        )
    plt.xlabel('fitted coherence')
    plt.ylabel('entropy estimate')
    plt.title('Empirical Entropy Estimates')
    plt.xticks(coherences, rotation=45)
    plt.tight_layout()
    
    plt.savefig('images_and_animations/entropies.png')
    # plt.show()

    coherence_test_idx = np.argmax(entropies)
    coherence_test = coherences[coherence_test_idx]
    coherence_test_arr = np.full_like(coherences, coherence_test)
    test_samples = np.zeros((fitness_samples.shape[0], num_samples_kstest))
    print("Collecting performance samples:")
    for n in tqdm(range(num_samples_kstest)):
        genomes = [Genome(x).update_randomstate() for x in genomes]
        results = client.map(evaluate, genomes, coherence_test_arr)
        results = client.gather(results)
        test_samples[:, n] = np.array(results)
    # for i in tqdm(range(test_samples.shape[0])):
    #     # genome = Genome(genomes[i]).update_randomstate()
    #     results = [
    #         # client.submit(evaluate, genome, coherence=c)
    #         evaluate(genome, coherence=coherence_test_idx)
    #         for sample in range(num_samples_kstest)
    #         ]
    #     test_samples[i, :] = np.array(results)
    print("Test samples:")
    print(test_samples)

    p_values = np.zeros(fitness_samples.shape[0])
    ks_statistics = np.zeros(fitness_samples.shape[0])
    for i in range(fitness_samples.shape[0]):
        mask = np.arange(fitness_samples.shape[0]) == i
        sample_1 = fitness_samples[i, :]
        sample_2 = fitness_samples[~mask, :].ravel()
        test_result = ks_2samp(sample_1, sample_2)
        p_values[i] = test_result.pvalue
        ks_statistics[i] = test_result.statistic
    # # to test that it works:
    # np.random.shuffle(ks_statistics)
    # plt.plot(ks_statistics)
    # plt.grid('off')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    
    alpha = 0.05
    reject_bool, p_values_corrected, alpha_sidak, alpha_bonf = multipletests(
        p_values, alpha=alpha, method='bonferroni',
        returnsorted=False, is_sorted=False)
    print("p-values | p-values corrected", *zip(p_values, p_values_corrected), sep='\n')

    plt.plot(
        coherences[reject_bool],
        p_values_corrected[reject_bool],
        '.', color='red', label='rejected'
        )
    plt.plot(
        coherences[~reject_bool],
        p_values_corrected[~reject_bool],
        'x', color='black', label='not rejected'
        )
    plt.axhline(alpha_bonf, label='bonferroni alpha', ls='--', color='black')
    plt.xlabel('fitted coherence')
    plt.ylabel('p-value')
    plt.yscale('log')
    plt.title(rf'Kolmogorov-Smirnov 2-Sample p-Values: Bonferonni $\alpha$={alpha_bonf:.4f}')
    plt.xticks(coherences, rotation=45)
    plt.grid(ls=':', alpha=0.1)
    plt.legend()
    plt.tight_layout()

    plt.savefig('images_and_animations/p_values.png')
    plt.show()
        # if i == 10:
        #     print(sample_1)
        #     print(sample_2)
        #     break
