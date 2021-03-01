# adapted from https://brian2.readthedocs.io/en/2.0rc/examples/synapses.STDP.html
import brian2 as b2
from brian2 import plt, np

taum = 10*b2.ms
taupre = 20*b2.ms
taupost = taupre
Ee = 0*b2.mV
vt = -54*b2.mV
vr = -60*b2.mV
El = -74*b2.mV
taue = 5*b2.ms
F = 15*b2.Hz 
w_max = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
# dApost = -dApre * taupre / taupost * 1.01
dApost *= w_max
dApre *= w_max

eqs_neurons = '''
dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
dge/dt = -ge / taue : 1
'''
eqs_synapses = '''
    w : 1
    mu : 1
    dApre/dt = -Apre / taupre : 1 (event-driven)
    dApost/dt = -Apost / taupost : 1 (event-driven)
    '''
on_pre = '''
    Apre += dApre * (w_max-w)**mu
    w = clip(w + Apost, 0, w_max)
    '''
on_post = '''
    Apost += dApost * w**mu
    w = clip(w + Apre, 0, w_max)
    '''
N_in = 2000
N_out = 100
N_bins = 100
default_rate = F * (1000/N_in)
postsyn_rates = b2.linspace(0.1, 99, N_out) * b2.Hz

if __name__ == '__main__':
    input_neurons = b2.PoissonGroup(
        N=N_in,
        rates=default_rate,
        name='inputgroup'
    )
    output_neurons_rates = b2.PoissonGroup(
        N=N_out,
        rates=postsyn_rates,
        name='outputgroup_rates'
    )
    output_neurons_mu = b2.NeuronGroup(
        N_out, eqs_neurons,
        threshold='v>vt',
        reset='v = vr',
        method='linear')
    # output_neurons_mu = b2.PoissonGroup(
    #     N=N_out,
    #     rates=default_rate,
    #     name='outputgroup_mu'
    # )
    
    synapse_rates = b2.Synapses(
        input_neurons, output_neurons_rates,
        eqs_synapses,
        on_pre=on_pre,
        on_post=on_post
    )
    synapse_rates.connect()
    synapse_rates.w = 'rand() * w_max'
    synapse_rates.mu = 0.

    synapse_mu = b2.Synapses(
        input_neurons, output_neurons_mu,
        eqs_synapses,
        on_pre='ge += w\n'+on_pre,
        on_post=on_post
    )
    synapse_mu.connect()
    synapse_mu.w = 'rand() * w_max'
    # for i in range(N_out):
    synapse_mu.mu = f'(j / {float(N_out)})'

    mon = b2.StateMonitor(synapse_mu, 'w', record=[0, 1])

    mu_vals = np.array(synapse_mu.mu)
    # print(mu_vals.shape)
    # b2.plot(mu_vals[synapse_mu.i == 0])
    # b2.show()

    net = b2.Network(b2.collect())
    net.store('initialised')  # not used...

    net.run(100*b2.second, report='stdout')

    b2.plot(mon.t/b2.second, mon.w.T/w_max)
    b2.show()

    weights_rates = b2.array(synapse_rates.w)
    rates_weight_histograms = b2.empty((N_bins, N_out))
    for i in range(N_out):
        # brian2 uses i->j synapse syntax, while I use j->i
        mask = synapse_rates.j == i
        rates_weight_histograms[:, i] = np.histogram(
            weights_rates[mask]/w_max,
            bins=N_bins,
            density=True,
            range=(0., 1.)
            )[0]

    weights_mu = b2.array(synapse_mu.w)
    mu_weight_histograms = b2.empty((N_bins, N_out))
    for i in range(N_out):
        # brian2 uses i->j synapse syntax, while I use j->i
        mask = synapse_mu.j == i
        mu_weight_histograms[:, i] = np.histogram(
            weights_mu[mask]/w_max,
            bins=N_bins,
            density=True,
            range=(0., 1.)
            )[0]

    b2.figure(figsize=(4,4))
    im = b2.imshow(
        rates_weight_histograms,
        origin='lower',
        extent=[postsyn_rates[0]/b2.Hz, postsyn_rates[-1]/b2.Hz, 0, 1],
        aspect='auto',
        cmap=b2.plt.cm.gray_r
        )
    # b2.colorbar(label='probability density')
    b2.xlabel("Postsynaptic Firing Rate (Hz)")
    # b2.ylabel(r"$\frac{w}{w_{max}}$", rotation=0)
    b2.ylabel(r"$w/w_{max}$")
    b2.title('Synaptic Strength Distributions\nas function of postsynaptic firing rate')
    b2.savefig('images_and_animations/synaptic_strengths_vs_firing_rates.png')
    # b2.show()

    b2.figure(figsize=(4,4))
    im = b2.imshow(
        mu_weight_histograms,
        origin='lower',
        extent=[0, 1, 0, 1],
        aspect='auto',
        cmap=b2.plt.cm.gray_r
        )
    # b2.colorbar(label='probability density')
    # b2.xticks(np.linspace(0., 1., 5)**2)
    b2.xlabel(r"$\mu$")
    # b2.ylabel(r"$\frac{w}{w_{max}}$", rotation=0)
    b2.ylabel(r"$w/w_{max}$")
    b2.title('Synaptic Strength Distributions')
    b2.savefig('images_and_animations/synaptic_strengths_vs_mu.png')
    b2.show()
