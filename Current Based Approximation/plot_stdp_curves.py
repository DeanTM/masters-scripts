import numpy as np
import matplotlib.pyplot as plt

@np.vectorize
def stdp(dt):
    if dt >= 0:
        return np.exp(-dt/10)
    else:
        return -np.exp(dt/10)

def istdp(dt):
    return -0.2 + np.abs(stdp(dt))

if __name__ == '__main__':
    dts = np.linspace(-40, 40, 100)
    fig, axes = plt.subplots(1,2, figsize=(10, 4), sharey=True)
    
    axes[0].plot(dts, stdp(dts), color='k')
    axes[1].plot(dts, istdp(dts), color='k')
    axes[0].set_title("Typical STDP Curve")
    axes[1].set_title("iSTDP Curve")

    axes[0].set_yticks([0.])
    axes[0].set_xticks([])
    axes[1].set_xticks([])
    axes[0].grid(ls=':', alpha=0.5)
    axes[1].grid(ls=':', alpha=0.5)
    axes[0].set_ylabel(r'$\Delta w$')
    axes[0].set_xlabel(r'$\Delta t$')
    axes[1].set_xlabel(r'$\Delta t$')
    fig.tight_layout()
    fig.savefig('images_and_animations/standard_stdpcurves.png')
    plt.show()