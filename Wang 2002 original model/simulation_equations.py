# A python file with differential equations for Wang 2002 simulation

#region Neuron Equations
neuron_eqns = '''
    I_syn = I_AMPA_ext + I_AMPA + I_NMDA + I_GABA : amp
    I_AMPA_ext = AMPA_CONDUCTANCE_EXT * s_AMPA_w_sum_ext * (u-V_E) : amp
    ds_AMPA_w_sum_ext/dt = -s_AMPA_w_sum_ext / tau_AMPA : 1
    
    I_AMPA = AMPA_CONDUCTANCE_REC * s_AMPA_w_sum_total * (u-V_E) : amp
    s_AMPA_w_sum_1 : 1
    s_AMPA_w_sum_2 : 1
    s_AMPA_w_sum_other : 1
    s_AMPA_w_sum_total = s_AMPA_w_sum_1 + s_AMPA_w_sum_2 + s_AMPA_w_sum_other : 1
    
    I_NMDA = NMDA_CONDUCTANCE * s_NMDA_w_sum_total * (u-V_E) / (1+Mg_conc*exp(-a_NMDA*u)/b_NMDA) : amp
    s_NMDA_w_sum_1 : 1
    s_NMDA_w_sum_2 : 1
    s_NMDA_w_sum_other : 1
    s_NMDA_w_sum_total = s_NMDA_w_sum_1 + s_NMDA_w_sum_2 + s_NMDA_w_sum_other : 1
    
    I_GABA = GABA_CONDUCTANCE * s_GABA_w_sum * (u-V_I) : amp
    s_GABA_w_sum : 1
    '''

neuron_eqns_membrane_noise = '''
    du/dt = (-(u-V_leak) - I_syn / LEAK_CONDUCTANCE) / TIMECONST + noise_membrane_std * xi / sqrt(TIMECONST) : volt (unless refractory)
    ''' + neuron_eqns

neuron_eqns = '''
    du/dt = (-(u-V_leak) - I_syn / LEAK_CONDUCTANCE) / TIMECONST : volt (unless refractory)
    ''' + neuron_eqns

neuron_eqns_excitatory_membrane_noise = neuron_eqns_membrane_noise.replace(
    'LEAK_CONDUCTANCE', 'g_E').replace(
    'TIMECONST', 'tau_mem_E').replace(
    'AMPA_CONDUCTANCE_EXT','g_AMPA_external_E').replace(
    'AMPA_CONDUCTANCE_REC', 'g_AMPA_E').replace(
    'NMDA_CONDUCTANCE', 'g_NMDA_E').replace(
    'GABA_CONDUCTANCE', 'g_GABA_E')
neuron_eqns_excitatory = neuron_eqns.replace(
    'LEAK_CONDUCTANCE', 'g_E').replace(
    'TIMECONST', 'tau_mem_E').replace(
    'AMPA_CONDUCTANCE_EXT','g_AMPA_external_E').replace(
    'AMPA_CONDUCTANCE_REC', 'g_AMPA_E').replace(
    'NMDA_CONDUCTANCE', 'g_NMDA_E').replace(
    'GABA_CONDUCTANCE', 'g_GABA_E')

neuron_eqns_inhibitory_membrane_noise = neuron_eqns_membrane_noise.replace(
    'LEAK_CONDUCTANCE', 'g_I').replace(
    'TIMECONST', 'tau_mem_I').replace(
    'AMPA_CONDUCTANCE_EXT','g_AMPA_external_I').replace(
    'AMPA_CONDUCTANCE_REC', 'g_AMPA_I').replace(
    'NMDA_CONDUCTANCE', 'g_NMDA_I').replace(
    'GABA_CONDUCTANCE', 'g_GABA_I')

neuron_eqns_inhibitory = neuron_eqns.replace(
    'LEAK_CONDUCTANCE', 'g_I').replace(
    'TIMECONST', 'tau_mem_I').replace(
    'AMPA_CONDUCTANCE_EXT','g_AMPA_external_I').replace(
    'AMPA_CONDUCTANCE_REC', 'g_AMPA_I').replace(
    'NMDA_CONDUCTANCE', 'g_NMDA_I').replace(
    'GABA_CONDUCTANCE', 'g_GABA_I')

threshold_eqn = 'u > V_thresh'
reset_eqn = 'u = V_reset'

#endregion

#region Synapse Equations
excitatory_synapse_variables = '''
    ds_AMPA/dt = -s_AMPA / tau_AMPA : 1 (clock-driven)
    dx_NMDA/dt = -x_NMDA / tau_x : 1 (clock-driven)
    ds_NMDA/dt = -s_NMDA / tau_NMDA + alpha_NMDA * x_NMDA * (1 - s_NMDA) : 1 (clock-driven)
    '''
inhibitory_synapse_variables = '''
    ds_GABA/dt = -s_GABA / tau_AMPA : 1 (clock-driven)
    '''

synapse_eqns_excitatory = '''
    w : 1
    s_NMDA_w_sum_SOURCE_post = w * s_NMDA : 1 (summed)
    s_AMPA_w_sum_SOURCE_post = w * s_AMPA : 1 (summed)
    ''' + excitatory_synapse_variables
synapse_eqns_excitatory_1 = synapse_eqns_excitatory.replace(
    'SOURCE', '1')
synapse_eqns_excitatory_2 = synapse_eqns_excitatory.replace(
    'SOURCE', '2')
synapse_eqns_excitatory_other = synapse_eqns_excitatory.replace(
    'SOURCE', 'other')
excitatory_on_pre = '''
    x_NMDA += 1
    s_AMPA += 1
    '''

synapse_eqns_inhibitory = '''
    w : 1
    s_GABA_w_sum_post = w * s_GABA : 1 (summed)
    ''' + inhibitory_synapse_variables
inhibitory_on_pre = '''
    s_GABA += 1
    '''
synapse_eqns_input = '''
    w : 1
    '''
input_on_pre = '''
    s_AMPA_w_sum_ext_post += w
    '''

#endregion

#region STDP equations
STDP_trace_eqns = '''
    dapre/dt = -apre/tau_pre : 1 (event-driven)
    dapost/dt = -apost/tau_post : 1 (event-driven)
    '''
STDP_on_pre = '''
    apre += A_pre_STDP
    w = clip(w+learning_rate*apost*(w**mu_weight), 0, w_max)
    '''
STDP_on_post = '''
    apost += A_post_STDP
    w = clip(w+learning_rate*apre*w, 0, w_max)
    '''

iSTDP_trace_eqns = '''
    dapre/dt = -apre/tau_iSTDP : 1 (event-driven)
    dapost/dt = -apost/tau_iSTDP : 1 (event-driven)
    '''
iSTDP_on_pre = '''
    apre += A_pre_iSTDP
    w = clip(w+learning_rate*(apost-alpha_iSTDP), 0, w_max)
    '''
iSTDP_on_post = '''
    apost += A_post_iSTDP
    w = clip(w+learning_rate*apre, 0, w_max)
    '''

#endregion
