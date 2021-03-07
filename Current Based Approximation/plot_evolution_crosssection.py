import numpy as np
import matplotlib.pyplot as plt

lr_samples_fname = 'experiments/plot_top_candidates_2021-02-28_19:12:06.488464/samples/samples.npy'
lr_coherences_fname = 'experiments/plot_top_candidates_2021-02-28_19:12:06.488464/samples/coherences.npy'

lr_meanstd_fname = 'run_evolution_crosssection_meanstd.npy'
lr_coherence_fname = 'run_evolution_crosssection_coherence.npy'
w_meanstd_fname = 'find_optimal_weights_crosssection_meanstd.npy'
w_coherence_fname = 'find_optimal_weights_crosssection_coherence.npy'

if __name__ == '__main__':
    lr_meanstd = np.load(lr_meanstd_fname)
    lr_coherences = np.load(lr_coherence_fname)
    w_meanstd = np.load(w_meanstd_fname) * 10
    w_coherences = np.load(w_coherence_fname)
    fitness_samples = np.load(lr_samples_fname)
    sample_coherences = np.load(lr_coherences_fname)

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        w_coherences, w_meanstd[:, 0], w_meanstd[:, 1],
        color='darkorange', linestyle='--', alpha=0.9,
        label='pop. avg. weights'
        )
    plt.errorbar(
        lr_coherences, lr_meanstd[:, 0], lr_meanstd[:, 1],
        color='gray', label='pop. avg. plasticity'
        )
    for i, c in enumerate(sample_coherences):
        x_vals = np.full_like(fitness_samples[i,:], c)
        label = None
        if i == 0:
            label = 'sampled performance'
        plt.plot(
            x_vals,
            fitness_samples[i,:], 'k.',
            label=label,
            markersize=4.,
            alpha=0.2
            )
    
    plt.legend()
    plt.xticks(sample_coherences, rotation=45)
    plt.xlabel('coherence')
    plt.ylabel('fitness (number of correct trials out of 100)')
    plt.title("Performances of Best Individuals and Generations")
    plt.grid(ls=':', alpha=0.2)
    plt.tight_layout()
    plt.savefig('images_and_animations/weight_lr_fitnesses_together.png')
    plt.show()
