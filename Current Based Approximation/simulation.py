from functions import *

# can't jit with **kwargs syntax
def F(
    W, nu,
    **weight_plasticity_params
):
    return np.outer(nu, nu)

tau_e_default = 0.1
# @jit(nopython=True)
def de_dt(
    W, eligibility_trace, nu,
    F_val=None,
    **weight_plasticity_params
):
    if F_val is None:
        F_val = F(W, nu, **weight_plasticity_params)
    tau_e = weight_plasticity_params.get('tau_e', tau_e_default)
    tau_de_dt = -eligibility_trace + F_val
    return tau_de_dt / tau_e
#     return np.zeros_like(eligibility_trace)

tau_w_default = 1.
beta_default = 0.
# @jit(nopython=True)
def dW_dt(
    W, eligibility_trace, nu, reward,  # TODO: take away nu input
    F_val=None,
    **weight_plasticity_params
):
    tau_w = weight_plasticity_params.get('tau_w', tau_w_default)
    beta = weight_plasticity_params.get('beta', beta_default)
    if F_val is None:
        if beta > 0.:
            F_val = F(W, nu, **weight_plasticity_params)
        else:
            F_val = 0.0
    tau_dw_dt = (1-beta)*eligibility_trace*reward + beta*F_val
    return tau_dw_dt / tau_w


tau_reward = defaultdt * 2
@jit(nopython=True)
def dR_dt(reward):
    dR_dt_now = (-reward)/defaultdt
    return dR_dt_now
    

def run_trial_coherence_2afc(
    initialisation_steps=10,
    lambda_=0.8,
    total_time=runtime,
    plasticity=True,
    W=None,
    coherence=0.5,
    trial_start=0.2,  # rename to stim_start
    trial_end=0.4,  # rename to stim_end
    eval_time=0.4,
    dR_dt=dR_dt,
    **weight_plasticity_params
):
    """Specifies stimulus and reward function for 2-afc task."""
    multiplier = np.ones(p+2)
    multiplier[1] += 0.05 * (1+coherence)
    multiplier[2] += 0.05 * (1-coherence)
    def reward_func(nu):
        reward = -1.0
        if np.argmax(nu[1:-1]) == 1:
            reward = 1.0
        return reward
        
    return run_trial(
        initialisation_steps=initialisation_steps,
        lambda_=lambda_,
        total_time=total_time,
        plasticity=plasticity,
        W=W,
        external_input_multiplier=multiplier,
        trial_start=trial_start,
        trial_end=trial_end,
        eval_time=eval_time,
        reward_func=reward_func,
        dR_dt=dR_dt,
        **weight_plasticity_params
    )


def run_trial(
    initialisation_steps=10,
    lambda_=0.8,
    total_time=runtime,
    plasticity=True,
    W=None,
    external_input_multiplier=np.ones(p+2),
    trial_start=0.2,  # rename to stim_start
    trial_end=0.4,  # rename to stim_end
    eval_time=0.4,
    reward_func=lambda x: 0.0,  # TODO
    dR_dt=dR_dt,
    **weight_plasticity_params
):
    C_k = np.array([N_non] + [N_sub] * p + [N_I])
    g_m = np.array([g_m_E] * (p+1) + [g_m_I])
    C_m = np.array([C_m_E] * (p+1) + [C_m_I])
    tau_m = np.array([tau_m_E] * (p+1) + [tau_m_I])
    tau_rp = np.array([tau_rp_E] * (p+1) + [tau_rp_I])
    nu = np.array([rate_pyramidal] * (p+1) + [rate_interneuron])
    
    # set weights
    if W is None:
        W = get_weights()
    else:
        assert W.shape[0] == p+2 
        W = W.copy()
    assert len(external_input_multiplier) == p+2
    
    # AMPA
    g_AMPA = np.array([g_AMPA_rec_E]* (p+1) + [g_AMPA_rec_I])
    s_AMPA = tau_AMPA * nu
    s_AMPA[~pyramidal_mask] = 0.  # inhibitory neurons won't feed AMPA-mediated synapses
    ip_AMPA = (V_drive - V_E) * C_k * s_AMPA
    ic_AMPA = g_AMPA * (W @ ip_AMPA)

    # AMPA_ext
    g_AMPA_ext = np.array([g_AMPA_ext_E]* (p+1) + [g_AMPA_ext_I])
    s_AMPA_ext = np.full_like(
        s_AMPA,
        tau_AMPA * rate_ext
    )  # array to allow for differing inputs
    ip_AMPA_ext = (V_drive - V_E) * C_ext * s_AMPA_ext
    ic_AMPA_ext = g_AMPA_ext * ip_AMPA_ext

    # GABA
    g_GABA = np.array([g_GABA_E]* (p+1) + [g_GABA_I])
    s_GABA = tau_GABA * nu
    s_GABA[pyramidal_mask] = 0.
    ip_GABA = (V_drive - V_I) * C_k * s_GABA
    ic_GABA = g_GABA * (W @  ip_GABA)

    # NMDA - requires self-consistent calculation
    g_NMDA = np.array([g_NMDA_E]* (p+1) + [g_NMDA_I])
    # can I speed these up with @numba.jit?
    g_NMDA_eff = lambda V: g_NMDA * J_2(V)
    V_E_eff = lambda V: V - (1 / J_2(V)) * (V - V_E) / J(V)

    s_NMDA = psi(nu)
    s_NMDA[~pyramidal_mask] = 0.
    
    # Initialise V_avg, V_SS
    # can I speed this up with @numba.jit?
    V_avg = np.full(p+2, V_avg_initial)
    for k in range(initialisation_steps):
        g_NMDA_eff_V = g_NMDA_eff(V_avg)
        V_E_eff_V = V_E_eff(V_avg)
        ip_NMDA = (V_drive - V_E_eff_V) * C_k * s_NMDA
        ic_NMDA = g_NMDA_eff_V * (W @ ip_NMDA)
        I_syn = ic_AMPA + ic_AMPA_ext + ic_NMDA + ic_GABA
        V_SS = V_L - I_syn / g_m
        V_avg = V_SS - (V_thr-V_reset)*nu*tau_m - (V_SS-V_reset)*nu*tau_rp
    
    # set sigma
    sigma = np.sqrt(
        g_AMPA_ext**2 * (V_drive - V_E)**2 * C_ext * rate_ext * tau_AMPA**2 / (g_m**2 * tau_m)
    )
    sigma[:-1] = lambda_ * (2e-3) + (1-lambda_) * sigma[:-1]
    
    ## Initialise arrays to track values, and for simulation
    times = np.arange(0, total_time, defaultdt)
    nu_tracked = np.full((p+2, times.shape[0]), np.nan)
    s_NMDA_tracked = np.full((p+2, times.shape[0]), np.nan)
    s_AMPA_tracked = np.full((p+2, times.shape[0]), np.nan)
    s_GABA_tracked = np.full((p+2, times.shape[0]), np.nan)
    I_syn_tracked = np.full((p+2, times.shape[0]), np.nan)
    e_tracked = np.full((p+2, p+2, times.shape[0]), np.nan)
    W_tracked = np.full((p+2, p+2, times.shape[0]), np.nan)
    
    if not plasticity:
        W_tracked = np.full((p+2, p+2, times.shape[0]), W.reshape(p+2, p+2, 1))
    
    ic_noise = np.zeros_like(s_AMPA)
    e = np.zeros_like(W)
    reward = 0.
    has_evaluated = False
    for itr, t in enumerate(times):
        ip_AMPA = (V_drive - V_E) * C_k * s_AMPA
        ic_AMPA = g_AMPA * (W @ ip_AMPA)

        g_NMDA_eff_V = g_NMDA_eff(V_avg)
        V_E_eff_V = V_E_eff(V_avg)
        ip_NMDA = (V_drive - V_E_eff_V) * C_k * s_NMDA
        ic_NMDA = g_NMDA_eff_V * (W @ ip_NMDA)

        ip_GABA = (V_drive - V_I) * C_k * s_GABA
        ic_GABA = g_GABA * (W @  ip_GABA)
        
        s_AMPA_ext = np.full_like(
            s_AMPA,
            tau_AMPA * rate_ext
        )
        if t > trial_start and t < trial_end:
            s_AMPA_ext = s_AMPA_ext * external_input_multiplier
            
        ip_AMPA_ext = (V_drive - V_E) * C_ext * s_AMPA_ext
        ic_AMPA_ext = g_AMPA_ext * ip_AMPA_ext
        sigma = np.sqrt(
            g_AMPA_ext**2 * (V_drive - V_E)**2 * C_ext * s_AMPA_ext * tau_AMPA / (g_m**2 * tau_m)
        )
        sigma[:-1] = lambda_ * (2e-3) + (1-lambda_) * sigma[:-1]
        
        if not has_evaluated and t >= eval_time:
            has_evaluated = True
            reward = reward_func(nu)
            
        
        I_syn = ic_AMPA + ic_AMPA_ext + ic_NMDA + ic_GABA + ic_noise
        dnu_dt_now = dnu_dt(
            nu, I_syn,
            sigma=sigma,
            g_m=g_m,
            tau_m=tau_m,
            tau_rp=tau_rp
        )
        dic_noise_dt_now = dic_noise_dt(ic_noise)
        ds_NMDA_dt_now = ds_NMDA_dt(s_NMDA, nu)
        ds_AMPA_dt_now = ds_AMPA_dt(s_AMPA, nu)
        ds_GABA_dt_now = ds_GABA_dt(s_GABA, nu)
        dreward_dt_now = dR_dt(reward)
        
        if plasticity:
            de_dt_now = de_dt(
                W, e, nu,
                **weight_plasticity_params
            )
            dW_dt_now = dW_dt(
                W, e, nu, reward,
                **weight_plasticity_params
            )
            
        nu += dnu_dt_now * defaultdt
        ic_noise += dic_noise_dt_now * defaultdt
        s_NMDA += ds_NMDA_dt_now * defaultdt
        s_AMPA += ds_AMPA_dt_now * defaultdt
        s_GABA += ds_GABA_dt_now * defaultdt
        reward += dreward_dt_now * defaultdt
        
        if plasticity:
            e += de_dt_now * defaultdt
            W += dW_dt_now * defaultdt
            W = W.clip(0.0, np.inf)    
            
        nu_tracked[:, itr] = nu
        s_NMDA_tracked[:, itr] = s_NMDA
        s_AMPA_tracked[:, itr] = s_AMPA
        s_GABA_tracked[:, itr] = s_GABA
        I_syn_tracked[:, itr] = I_syn
        
        if plasticity:
            e_tracked[:, :, itr] = e
            W_tracked[:, :, itr] = W
            
        # Not mentioned in W&W2006:
        V_SS = V_L - I_syn / g_m
        V_avg = V_SS - (V_thr-V_reset)*nu*tau_m - (V_SS-V_reset)*nu*tau_rp

        # TODO: for fitness, return some default or penalised score??
        if np.any(np.isnan(nu)):
            break
        
    return  times,\
        nu_tracked,\
        s_NMDA_tracked,\
        s_AMPA_tracked,\
        s_GABA_tracked,\
        I_syn_tracked,\
        e_tracked,\
        W_tracked
