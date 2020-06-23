# A python file with config parameters for Wang 2002 simulation
import brian2 as b2

# default parameters to be used by Parameters class
NETWORK_PARAMS = dict(
    N_E = 1600,
    N_I = 400,
    f_sel = 0.15,
    w_plus = 1.7,
    N_choices = 2
)
NETWORK_PARAMS['N_sel'] = int(NETWORK_PARAMS['f_sel'] * NETWORK_PARAMS['N_E'])

INPUT_PARAMS = dict(
    update_dt = 50.0 * b2.ms,
    coherence_level = 0.3,
    stimulus_mean = 40.0 * b2.Hz,
)
INPUT_PARAMS['lookback'] = int(INPUT_PARAMS['update_dt']/ b2.defaultclock.dt)
INPUT_PARAMS['N_input'] = NETWORK_PARAMS['N_sel']

NEURON_SYNAPSE_PARAMS = dict(
    V_leak = -70 * b2.mV,
    V_thresh = -50 * b2.mV,
    V_reset = -55 * b2.mV,
    V_E = 0 * b2.mV,
    V_I = -70 * b2.mV,
    tau_mem_E = 20 * b2.ms,
    tau_mem_I = 10 * b2.ms,
    tau_refrac_E = 2 * b2.ms,
    tau_refrac_I = 1 * b2.ms,
    tau_AMPA = 2 * b2.ms,
    tau_GABA = 5 * b2.ms,
    tau_NMDA = 100 * b2.ms,
    tau_x = 2 * b2.ms ,
    synapse_delay = 0.5 * b2.ms,
    alpha_NMDA = 0.5 * b2.kHz,
    a_NMDA = 0.062 * b2.mV ** -1,
    b_NMDA = 3.57,
    Mg_conc = 1 ,
    g_E = 25 * b2.nS,
    g_I = 20 * b2.nS,
    g_AMPA_external_E = 2.1 * b2.nS,
    g_AMPA_E = 80 * b2.nS / NETWORK_PARAMS['N_E'],
    g_NMDA_E = 264 * b2.nS / NETWORK_PARAMS['N_E'] ,
    g_GABA_E = 520 * b2.nS / NETWORK_PARAMS['N_I'],
    g_AMPA_external_I = 1.62 * b2.nS,
    g_AMPA_I = 64 * b2.nS / NETWORK_PARAMS['N_E'],
    g_NMDA_I = 208 * b2.nS / NETWORK_PARAMS['N_E'],
    g_GABA_I = 400 * b2.nS / NETWORK_PARAMS['N_I']
)

PLASTICITY_PARAMS = dict(
    learning_rate = 1e-2,
    tau_pre = 20 * b2.ms,
    tau_post = 20 * b2.ms,
    A_pre_STDP = 0.01,
    mu_weight = 0.023,
    tau_iSTDP = 20 * b2.ms,
    A_pre_iSTDP = 1.0,
    A_post_iSTDP = 1.0,
    use_STDP = False,
    use_iSTDP = False
)
PLASTICITY_PARAMS['A_post_STDP'] = -PLASTICITY_PARAMS['A_pre_STDP'] * 1.05
PLASTICITY_PARAMS['alpha_iSTDP'] = 3 * b2.Hz * PLASTICITY_PARAMS['tau_iSTDP'] * 2

DEFAULT_PARAMS = NETWORK_PARAMS.copy()
DEFAULT_PARAMS.update(INPUT_PARAMS)
DEFAULT_PARAMS.update(NEURON_SYNAPSE_PARAMS)
DEFAULT_PARAMS.update(PLASTICITY_PARAMS)
