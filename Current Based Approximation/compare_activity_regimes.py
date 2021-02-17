import brian2 as b2
from brian2 import plt
from parameters_spiking import *


def simulate(runtime=0.5*b2.second, N=1):
    b2.start_scope()

    namespace['sigma'] = 2 * b2.mV
    namespace['tau_m_E'] = namespace['C_m_E'] / namespace['g_m_E']
    # I_1 = -2*b2.namp
    # I_2 = -20*b2.namp
    I_1 = namespace['g_m_E'] * (namespace['V_L'] - (2.5*b2.mV + namespace['V_thr']))
    I_2 = namespace['g_m_E'] * (namespace['V_L'] - (-2.5*b2.mV + namespace['V_thr']))

    eqn = """
    dV/dt = (- g_m_E * (V - V_L) - I) / C_m_E + sigma*xi*tau_m_E**-0.5: volt (unless refractory)
    I : amp
    """
    N1 = b2.NeuronGroup(
        N, eqn, threshold='V>V_thr',
        reset='V = V_reset',
        refractory=namespace['tau_rp_E'],
        method='euler')
    N1.V = namespace['V_reset']
    N1.I = I_1

    N2 = b2.NeuronGroup(
        N, eqn, threshold='V>V_thr',
        reset='V = V_reset',
        refractory=namespace['tau_rp_E'],
        method='euler')
    N2.V = namespace['V_reset']
    N2.I = I_2
    
    st1 = b2.StateMonitor(N1, variables='V', record=True, name='st1')
    st2 = b2.StateMonitor(N2, variables='V', record=True, name='st2')
    sp1 = b2.SpikeMonitor(N1, name='sp1')
    sp2 = b2.SpikeMonitor(N2, name='sp2')

    net = b2.Network(b2.collect())
    net.run(runtime, namespace=namespace, report='stdout')
    return net


if __name__ == "__main__":
    N = 2
    net = simulate(runtime=60*b2.second, N=N)

    max_time = 0.7 * b2.second
    min_time = 0.2 * b2.second

    # fig, axes = plt.subplots(2,2,figsize=(12, 8))
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2,2,1)
    mask = (net['st1'].t > min_time) & (net['st1'].t < max_time)
    # lines = axes[0,0].plot(net['st1'].t[mask], net['st1'].V.T[mask, :]/b2.mV, alpha=0.8, lw=1.)
    lines = ax1.plot(net['st1'].t[mask], net['st1'].V.T[mask, :]/b2.mV, alpha=0.8, lw=1.)
    for t,i in zip(net['sp1'].t, net['sp1'].i):
        if t > min_time and t < max_time:
            # axes[0,0].axvline(t/b2.second, color=lines[i].get_color(), ls=':', alpha=.3)
            ax1.axvline(t/b2.second, color=lines[i].get_color(), ls=':', alpha=.3)
    # axes[0,0].set_ylabel('membrane potential (mV)\nmean-drive regime')
    # axes[0,0].set_title('Membrane Potential Trajectories')
    ax1.set_ylabel('membrane potential (mV)\nmean-drive regime')
    ax1.set_title('Membrane Potential Trajectories')
    ax1.set_xlim(min_time/b2.second, max_time/b2.second)
    ax1.grid(ls=':', alpha=0.2)

    ax2 = plt.subplot(2,2,2)
    for i in range(N):
        # axes[0,1].hist(b2.diff(net['sp1'].t[net['sp1'].i == i])/b2.second, bins=30)
        ax2.hist(b2.diff(net['sp1'].t[net['sp1'].i == i])/b2.second, bins=30, alpha=0.8)
    # axes[0,1].set_title('Histograms of Inter-Spike Intervals (ISIs)')
    ax2.set_title('Histograms of Inter-Spike Intervals (ISIs)')
    ax2.grid(ls=':', alpha=0.2)

    ax3 = plt.subplot(2,2,3, sharex=ax1)
    mask = (net['st2'].t > min_time) & (net['st2'].t < max_time)
    # lines = axes[1,0].plot(net['st2'].t[mask], net['st2'].V.T[mask, :]/b2.mV, alpha=0.8, lw=1.)
    lines = ax3.plot(net['st2'].t[mask], net['st2'].V.T[mask, :]/b2.mV, alpha=0.8, lw=1.)
    for t,i in zip(net['sp2'].t, net['sp2'].i):
        if t > min_time and t < max_time:
            # axes[1,0].axvline(t/b2.second, color=lines[i].get_color(), ls=':', alpha=.3)
            ax3.axvline(t/b2.second, color=lines[i].get_color(), ls=':', alpha=.3)
    # axes[1,0].set_ylabel('membrane potential (mV)\nfluctuation-drive regime')
    # axes[1,0].set_xlabel('time')
    ax3.set_ylabel('membrane potential (mV)\nfluctuation-drive regime')
    ax3.set_xlabel('time')
    ax3.grid(ls=':', alpha=0.2)

    ax4 = plt.subplot(2,2,4, sharex=ax2)
    for i in range(N):
        # axes[1,1].hist(b2.diff(net['sp2'].t[net['sp2'].i == i])/b2.second, bins=30)
        ax4.hist(b2.diff(net['sp2'].t[net['sp2'].i == i])/b2.second, bins=30, alpha=0.8)
    # axes[1,1].set_xlabel('ISI')
    ax4.set_xlabel('ISI')
    ax4.grid(ls=':', alpha=0.2)

    plt.tight_layout()
    plt.savefig('comparison-of-firing-regimes.png')
    plt.show()


