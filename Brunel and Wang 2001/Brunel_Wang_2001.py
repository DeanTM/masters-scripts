# code taken from brian2 examples list, and edited
from brian2 import *
from brian2 import numpy as np
import argparse

description = """
This script runs the Brunel and Wang 2001 n-alternative decision making model 
with the Brian2 simulator. 

The model will likely be extended to include plasticity and edited to work 
with reinforcement learning environments.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-N', '--neurons',
    help="total number of neurons",
    type=int, default=1000)
parser.add_argument('-p', '--populations',
    help="total number of selective",
    type=int, default=5)
parser.add_argument('-f', '--fraction',
    help="fraction of excitatory neurons in selective population",
    type=float, default=0.1)
parser.add_argument('-s', '--sigma',
    help="standard deviation for normal weights",
    type=float, default=0.0)
parser.add_argument('-w', '--w_plus',
    help="mean recurrent selective group weights",
    type=float, default=2.1)
parser.add_argument('-T', '--runtime',
    help="total simulation runtime (seconds)",
    type=float, default=4.0)
parser.add_argument('-t', '--stimtime',
    help="total selective stimulation time (milliseconds) rounded to nearest multiple of 25ms",
    type=int, default=50)
parser.add_argument('-r', '--rate',
    help="firing rate for selected population (hertz)",
    type=float, default=25.0)
parser.add_argument('--STDP',
    help="whether to enable STDP at excitatory synapses",
    action="store_true")
parser.add_argument('-a', '--Apre',
    help="LTP increment size for STDP rule",
    type=float, default=0.01)
parser.add_argument('--seed',
    help="random seed for Brian2",
    type=int)
# TODO: add plotting arguments
# TODO: add W-STDP and Triplet Rule simulations

args = parser.parse_args()

print(args)
# for k, v in vars(args).items():
#     print(f"{k}: {v}")
assert args.fraction <= 1.0 and args.fraction >= 0, \
    'selective fraction must be between 0 and 1'
assert args.fraction * args.populations <= 1.0, \
    'selective fraction does not support number of populations'
if args.seed:
    seed(args.seed)

# if args.version:
#     print("This is version 0.1")

runtime = args.runtime * second

# populations
# N = 1000
N = args.neurons
N_E = int(N * 0.8)  # pyramidal neurons
N_I = int(N * 0.2)  # interneurons

simulation_namespace = dict()

# voltage
# V_L = -70. * mV
# V_thr = -50. * mV
# V_reset = -55. * mV
# V_E = 0. * mV
# V_I = -70. * mV
simulation_namespace.update(dict(
    V_L = -70. * mV,
    V_thr = -50. * mV,
    V_reset = -55. * mV,
    V_E = 0. * mV,
    V_I = -70. * mV
))

# membrane capacitance
# C_m_E = 0.5 * nF
# C_m_I = 0.2 * nF
simulation_namespace.update(dict(
    C_m_E = 0.5 * nF,
    C_m_I = 0.2 * nF
))

# membrane leak
# g_m_E = 25. * nS
# g_m_I = 20. * nS
simulation_namespace.update(dict(
    g_m_E = 25. * nS,
    g_m_I = 20. * nS
))

# refractory period
# tau_rp_E = 2. * ms
# tau_rp_I = 1. * ms
simulation_namespace.update(dict(
    tau_rp_E = 2. * ms,
    tau_rp_I = 1. * ms
))

# external stimuli
rate_ext = 3 * Hz
C_ext = 800

# synapses
C_E = N_E
C_I = N_I

# AMPA (excitatory)
# g_AMPA_ext_E = 2.08 * nS
# g_AMPA_rec_E = 0.104 * nS * 800. / N_E
# g_AMPA_ext_I = 1.62 * nS
# g_AMPA_rec_I = 0.081 * nS * 800. / N_E
# tau_AMPA = 2. * ms
simulation_namespace.update(dict(
    g_AMPA_ext_E = 2.08 * nS,
    g_AMPA_rec_E = 0.104 * nS * 800. / N_E,
    g_AMPA_ext_I = 1.62 * nS,
    g_AMPA_rec_I = 0.081 * nS * 800. / N_E,
    tau_AMPA = 2. * ms
))

# NMDA (excitatory)
# g_NMDA_E = 0.327 * nS * 800. / N_E
# g_NMDA_I = 0.258 * nS * 800. / N_E
# tau_NMDA_rise = 2. * ms
# tau_NMDA_decay = 100. * ms
# alpha = 0.5 / ms
# Mg2 = 1.
simulation_namespace.update(dict(
    g_NMDA_E = 0.327 * nS * 800. / N_E,
    g_NMDA_I = 0.258 * nS * 800. / N_E,
    tau_NMDA_rise = 2. * ms,
    tau_NMDA_decay = 100. * ms,
    alpha = 0.5 / ms,
    Mg2 = 1.
))

# GABAergic (inhibitory)
# g_GABA_E = 1.25 * nS * 200. / N_I
# g_GABA_I = 0.973 * nS * 200. / N_I
# tau_GABA = 10. * ms
simulation_namespace.update(dict(
    g_GABA_E = 1.25 * nS * 200. / N_I,
    g_GABA_I = 0.973 * nS * 200. / N_I,
    tau_GABA = 10. * ms
))
    
# subpopulations
# f = 0.1
f = args.fraction
# p = 5
p = args.populations
N_sub = int(N_E * f)
N_non = int(N_E * (1. - f * p))
# w_plus = 2.1
w_plus = args.w_plus
w_minus = 1. - f * (w_plus - 1.) / (1. - f)
w_std = args.sigma

# modeling
eqs_E = '''
dv / dt = (- g_m_E * (v - V_L) - I_syn) / C_m_E : volt (unless refractory)

I_syn = I_AMPA_ext + I_AMPA_rec + I_NMDA_rec + I_GABA_rec : amp

I_AMPA_ext = g_AMPA_ext_E * (v - V_E) * s_AMPA_ext : amp
I_AMPA_rec = g_AMPA_rec_E * (v - V_E) * 1 * s_AMPA : amp
ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA : 1
ds_AMPA / dt = - s_AMPA / tau_AMPA : 1

I_NMDA_rec = g_NMDA_E * (v - V_E) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) * s_NMDA_tot : amp
s_NMDA_tot : 1

I_GABA_rec = g_GABA_E * (v - V_I) * s_GABA : amp
ds_GABA / dt = - s_GABA / tau_GABA : 1
'''

eqs_I = '''
dv / dt = (- g_m_I * (v - V_L) - I_syn) / C_m_I : volt (unless refractory)

I_syn = I_AMPA_ext + I_AMPA_rec + I_NMDA_rec + I_GABA_rec : amp

I_AMPA_ext = g_AMPA_ext_I * (v - V_E) * s_AMPA_ext : amp
I_AMPA_rec = g_AMPA_rec_I * (v - V_E) * 1 * s_AMPA : amp
ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA : 1
ds_AMPA / dt = - s_AMPA / tau_AMPA : 1

I_NMDA_rec = g_NMDA_I * (v - V_E) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) * s_NMDA_tot : amp
s_NMDA_tot : 1

I_GABA_rec = g_GABA_I * (v - V_I) * s_GABA : amp
ds_GABA / dt = - s_GABA / tau_GABA : 1
'''

P_E = NeuronGroup(N_E, eqs_E, threshold='v > V_thr', reset='v = V_reset', refractory='tau_rp_E', method='euler')
P_E.v = simulation_namespace['V_L']
P_I = NeuronGroup(N_I, eqs_I, threshold='v > V_thr', reset='v = V_reset', refractory='tau_rp_I', method='euler')
P_I.v = simulation_namespace['V_L']

eqs_glut = '''
s_NMDA_tot_post = w * s_NMDA : 1 (summed)
ds_NMDA / dt = - s_NMDA / tau_NMDA_decay + alpha * x * (1 - s_NMDA) : 1 (clock-driven)
dx / dt = - x / tau_NMDA_rise : 1 (clock-driven)
w : 1
'''

eqs_pre_glut = '''
s_AMPA += w
x += 1
'''

eqs_pre_gaba = '''
s_GABA += 1
'''

eqs_pre_ext = '''
s_AMPA_ext += 1
'''

# STDP
eqs_post_glut = ''
if args.STDP:
    # Taken mostly from Brian2 STDP example
    taupre = taupost = 20*ms
    simulation_namespace.update(dict(
        taupre = taupre,
        taupost = taupost,
        wmax = 5,
        Apre = args.Apre,
        Apost = -args.Apre*taupre/taupost*1.05
    ))
    eqs_glut += '''
    dapre/dt = -apre/taupre : 1 (event-driven)
    dapost/dt = -apost/taupost : 1 (event-driven)
    '''
    eqs_pre_glut += '''
    apre += Apre
    w = clip(w+apost, 0, wmax)
    '''
    eqs_post_glut = '''
    apost += Apost
    w = clip(w+apre, 0, wmax)
    '''

def get_lognormal_weights(target_mean, num_weights, normal_std=0.0):
    input_mean = np.log(target_mean) - 0.5*normal_std**2
    return np.random.lognormal(
        mean=input_mean, sigma=normal_std, size=num_weights
    )

# E to E
C_E_E = Synapses(P_E, P_E, 
    model=eqs_glut, on_pre=eqs_pre_glut, on_post=eqs_post_glut,
    method='euler')
C_E_E.connect('i != j')
# C_E_E.w[:] = 1
num_weights = C_E_E.w[:].shape[0]
C_E_E.w[:] = get_lognormal_weights(
    target_mean=1.0,
    num_weights=num_weights,
    normal_std=0.0
)

for pi in range(N_non, N_non + p * N_sub, N_sub):
    # internal other subpopulation to current nonselective
    # C_E_E.w[C_E_E.indices[:, pi:pi + N_sub]] = w_minus
    num_weights = C_E_E.indices[:, pi:pi + N_sub].shape[0]
    C_E_E.w[C_E_E.indices[:, pi:pi + N_sub]] = get_lognormal_weights(
        target_mean=w_minus,
        num_weights=num_weights,
        normal_std=0.0
    )
    # internal current subpopulation to current subpopulation
    # C_E_E.w[C_E_E.indices[pi:pi + N_sub, pi:pi + N_sub]] = w_plus
    num_weights = C_E_E.indices[pi:pi + N_sub, pi:pi + N_sub].shape[0]
    C_E_E.w[C_E_E.indices[pi:pi + N_sub, pi:pi + N_sub]] = get_lognormal_weights(
        target_mean=w_plus,
        num_weights=num_weights,
        normal_std=0.0
    )

# E to I
C_E_I = Synapses(P_E, P_I, model=eqs_glut, on_pre=eqs_pre_glut, method='euler')
C_E_I.connect()
C_E_I.w[:] = 1

# I to I
C_I_I = Synapses(P_I, P_I, on_pre=eqs_pre_gaba, method='euler')
C_I_I.connect('i != j')

# I to E
C_I_E = Synapses(P_I, P_E, on_pre=eqs_pre_gaba, method='euler')
C_I_E.connect()

# external noise
C_P_E = PoissonInput(P_E, 's_AMPA_ext', C_ext, rate_ext, '1')
C_P_I = PoissonInput(P_I, 's_AMPA_ext', C_ext, rate_ext, '1')

# at 1s, select population 1
C_selection = int(f * C_ext)
# rate_selection = 25 * Hz
rate_selection = args.rate * Hz
# stimuli1 = TimedArray(np.r_[np.zeros(40), np.ones(2), np.zeros(100)], dt=25 * ms)
stimtime = int(args.stimtime / 25)
stimuli1 = TimedArray(np.r_[np.zeros(40), np.ones(stimtime), np.zeros(100)], dt=25 * ms)
simulation_namespace['stimuli1'] = stimuli1
input1 = PoissonInput(P_E[N_non:N_non + N_sub], 's_AMPA_ext', C_selection, rate_selection, 'stimuli1(t)')

# at 2s, select population 2
# stimuli2 = TimedArray(np.r_[np.zeros(80), np.ones(2), np.zeros(100)], dt=25 * ms)
# simulation_namespace['stimuli2'] = stimuli2
# input2 = PoissonInput(P_E[N_non + N_sub:N_non + 2 * N_sub], 's_AMPA_ext', C_selection, rate_selection, 'stimuli2(t)')

# at 4s, reset selection
# stimuli_reset = TimedArray(np.r_[np.zeros(120), np.ones(2), np.zeros(100)], dt=25 * ms)
# simulation_namespace['stimuli_reset'] = stimuli_reset
# input_reset_I = PoissonInput(P_E, 's_AMPA_ext', C_ext, rate_selection, 'stimuli_reset(t)')
# input_reset_E = PoissonInput(P_I, 's_AMPA_ext', C_ext, rate_selection, 'stimuli_reset(t)')

# monitors
N_activity_plot = 15
sp_E_sels = [SpikeMonitor(P_E[pi:pi + N_activity_plot]) for pi in range(N_non, N_non + p * N_sub, N_sub)]
sp_E = SpikeMonitor(P_E[:N_activity_plot])
sp_I = SpikeMonitor(P_I[:N_activity_plot])

r_E_sels = [PopulationRateMonitor(P_E[pi:pi + N_sub]) for pi in range(N_non, N_non + p * N_sub, N_sub)]
r_E = PopulationRateMonitor(P_E[:N_non])
r_I = PopulationRateMonitor(P_I)

if args.STDP:
    W_before = np.full((len(P_E), len(P_E)), np.nan)
    # Insert the values from the Synapses object
    W_before[C_E_E.i[:], C_E_E.j[:]] = C_E_E.w[:]

# simulate, can be long >120s
net = Network(collect())
net.add(sp_E_sels)
net.add(r_E_sels)
# net.run(4 * second, report='stdout')
net.run(runtime,
    report='stdout',
    namespace=simulation_namespace)

if args.STDP:
    W_after = np.full((len(P_E), len(P_E)), np.nan)
    # Insert the values from the Synapses object
    W_after[C_E_E.i[:], C_E_E.j[:]] = C_E_E.w[:]

# plotting
title('Population rates')
xlabel('ms')
ylabel('Hz')

plot(r_E.t / ms, r_E.smooth_rate(width=25 * ms) / Hz, label='nonselective')
plot(r_I.t / ms, r_I.smooth_rate(width=25 * ms) / Hz, label='inhibitory')

for i, r_E_sel in enumerate(r_E_sels[::-1]):
    plot(r_E_sel.t / ms, r_E_sel.smooth_rate(width=25 * ms) / Hz, label='selective {}'.format(p - i))

legend()
figure()

title('Population activities ({} neurons/pop)'.format(N_activity_plot))
xlabel('ms')
yticks([])

plot(sp_E.t / ms, sp_E.i + (p + 1) * N_activity_plot, '.', markersize=2, label='nonselective')
plot(sp_I.t / ms, sp_I.i + p * N_activity_plot, '.', markersize=2, label='inhibitory')

for i, sp_E_sel in enumerate(sp_E_sels[::-1]):
    plot(sp_E_sel.t / ms, sp_E_sel.i + (p - i - 1) * N_activity_plot, '.', markersize=2, label='selective {}'.format(p - i))

legend()

if args.STDP:
    fig, axes = subplots(1, 2)
    axes[0].imshow(W_before, vmin=0.0, vmax=simulation_namespace['wmax'])
    axes[1].imshow(W_after, vmin=0.0, vmax=simulation_namespace['wmax'])
    axes[0].set_title('Weights Before')
    axes[1].set_title('Weights After')
    
show()
