# A file to create Simulation class instances, which can run the simulation
# with it's own parameters
import brian2 as b2
from brian2 import numpy as np


lif_membrane_eqn = '''
    du/dt = (-(u-REST_POTENTIAL) - I_syn / LEAK_CONDUCTANCE) / TIMECONST + noise_membrane_std * xi / sqrt(TIMECONST) : volt (unless refractory)
    '''
lif_membrane_eqn_nonoise = '''
    du/dt = (-(u-REST_POTENTIAL) - I_syn / LEAK_CONDUCTANCE) / TIMECONST : volt (unless refractory)
    '''

# not (REVERSAL_POTENTIAL_EXCITE - u) because -I_syn is used i.e. negative 
# current flows out of the synapse
conductance_synapse_eqns = '''
    I_syn = I_excite + I_inhibit : amp
    I_excite = EXCITE_CONDUCTANCE * s_excite * (u - REVERSAL_POTENTIAL_EXCITE) : amp
    ds_excite/dt = -s_excite / EXCITE_TIMECONST : 1  # incremented by input and other excitatory populations
    I_inhibit = INHIBIT_CONDUCTANCE * s_inhibit * (u - REVERSAL_POTENTIAL_INHIBIT) : amp
    ds_inhibit/dt = -s_inhibit / INHIBIT_TIMECONST : 1 
    '''

# specify synaptic strength and continuous weight updates here
synapse_eqns = '''
    w : 1
    '''
synapse_excite_on_pre = '''
    s_excite_post += w
    '''
synapse_inhibit_on_pre = '''
    s_inhibit_post += w
    '''
# x_i traces are of presynaptic activity, y_i traces are of postsynaptic
# actvity. ee/ie indicates synapse type.
# If we give the namespace when specifying the synapse, we don't need to
# specify whether the timeconst. is ee or ie
NUM_TRACES = 2
synpatic_trace_eqns_ee = ''.join([
    f'''
    dx_{i}/dt = -x_{i}/TIMECONST_x_{i}_ee : 1 (event-driven)
    dy_{i}/dt = -y_{i}/TIMECONST_y_{i}_ee : 1 (event-driven)
    '''
    for i in range(1, NUM_TRACES+1)])
synpatic_trace_eqns_ie = ''.join([
    f'''
    dx_{i}/dt = -x_{i}/TIMECONST_x_{i}_ie : 1 (event-driven)
    dy_{i}/dt = -y_{i}/TIMECONST_y_{i}_ie : 1 (event-driven)
    '''
    for i in range(1, NUM_TRACES+1)])


# TODO: This ignores trace-trace multiplicative interactions!!
# presynaptic spikes increment weights by postsynaptic amounts
weight_increment_ee_on_pre = '\nincrement = ' + ' + '.join([
    f'y_{i}*w**WEIGHT_EXP_y_{i}'
    for i in range(1, NUM_TRACES+1)])
synaptic_traces_ee_on_pre = '\n'.join([
    f'x_{i} += Apre_{i}_ee'
    for i in range(1, NUM_TRACES+1)])\
    + weight_increment_ee_on_pre\
    + '\nw = clip(w + LEARNING_RATE_EE * increment, W_MIN, W_MAX)'

# postsynaptic spikes increment weights by presynaptic amounts
weight_increment_ee_on_post = '\nincrement = ' + ' + '.join([
    f'x_{i}*w**WEIGHT_EXP_x_{i}'
    for i in range(1, NUM_TRACES+1)])
synaptic_traces_ee_on_post = '\n'.join([
    f'y_{i} += Apost_{i}_ee'
    for i in range(1, NUM_TRACES+1)])\
    + weight_increment_ee_on_post\
    + '\nw = clip(w + LEARNING_RATE_EE * increment, W_MIN, W_MAX)'
print(synaptic_traces_ee_on_post)

# params for reference
A_STDP_params = dict(
    LEARNING_RATE_EE = 1,
    W_MIN = 0,  # Dale's law
    W_MAX = 10,  # or any upper bound
    # additive STDP has no weight dependence
    WEIGHT_EXP_x_1 = 0,
    WEIGHT_EXP_x_2 = 0,
    WEIGHT_EXP_y_1 = 0,
    WEIGHT_EXP_y_2 = 0,
    # only one pre and one post synaptic trace
    Apre_1 = 1,  # or any increment size
    Apre_2 = 0,
    Apost_1 = -1,  # or any decrement size
    Apost_2 = 0,
)




class Simulation():
    """An instance of a Brian2 simulation which can be run to evaluate parameter 
    fitness.
    """
    def __init__(self, *args, **kwargs):
        self.network = None
        pass

    def prepare_network(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        if self.network is None:
            self.prepare_network(*args, **kwargs)
        pass
