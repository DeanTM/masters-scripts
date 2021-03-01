import matplotlib.pyplot as plt
import numpy as np
from functions import psi


if __name__ == '__main__':
    nu = np.linspace(0, 80, 20)
    truncations = [1, 2, 3, 5, 10, 20]

    most_accurate_truncation = 100
    most_accurate = psi(nu, n_truncate=most_accurate_truncation)
    # most_accurate = []
    # for hz in nu:
    #     most_accurate.append(psi(hz, n_truncate=most_accurate_truncation))
    most_accurate = np.array(most_accurate)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    max_abs_deviations = []
    for n_truncate in truncations:
        s_NMDA_vals = psi(nu, n_truncate=n_truncate)
        # s_NMDA_vals = []
        # for hz in nu:
        #     s_NMDA_vals.append(psi(hz, n_truncate=n_truncate))
        s_NMDA_vals = np.array(s_NMDA_vals)
        max_abs_deviations.append(np.max(
            np.abs(s_NMDA_vals - most_accurate)
        ))
        axes[0].plot(nu, s_NMDA_vals, label=n_truncate)
        
    axes[0].plot(nu, most_accurate, 'k:', label=most_accurate_truncation)
    axes[0].grid(ls=':')
    axes[0].legend(title='$n$ truncation')
    axes[0].set_ylim([0.0, 1.0])
    axes[0].set_xlim([0.0, 80.0])
    axes[0].set_title(r'asymptotic $\langle s_{NMDA} \rangle$' + "\n" + r'as function of presynaptic rate $\nu_k$')
    axes[0].set_xlabel(r'$\nu_k$ (Hz)')
    axes[0].set_ylabel('Fraction of open NMDA channels')
    axes[1].plot(truncations, max_abs_deviations)
    axes[1].set_xticks(truncations)
    axes[1].set_xlabel('truncation value $n$')
    axes[1].set_ylabel('max absolute deviation')
    axes[1].set_title(f'maximum deviation from \n$\psi$ computed with $n={most_accurate_truncation}$')
    plt.tight_layout()
    fig.savefig('images_and_animations/psi_truncation.png')
    plt.show()