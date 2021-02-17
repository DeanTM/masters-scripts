from functions import *
from collections.abc import Iterable
from functools import partial


param_names = [
    'p_const', 'p_theta', 'mu', 'tau_theta','xi_00', 
    'xi_10_0', 'xi_10_1', 'xi_01_0', 'xi_01_1', 'xi_11_0', 'xi_11_1',
    'xi_20_0', 'xi_20_1', 'xi_21_0', 'xi_21_1', 'xi_02_0', 'xi_02_1',
    'xi_12_0', 'xi_12_1', 'tau_e', 'beta'
    ]
param_names_latex = [
    r'$p_{const}$', r'$p_{\theta}$', r'$\mu$', r'$\tau_{\theta}$',r'$\xi_{00}$', 
    r'$\xi^{10}_0$', r'$\xi^{10}_1$', r'$\xi^{01}_0$', r'$\xi^{01}_1$', r'$\xi^{11}_0$', r'$\xi^{11}_1$',
    r'$\xi^{20}_0$', r'$\xi^{20}_1$', r'$\xi^{21}_0$', r'$\xi^{21}_1$', r'$\xi^{02}_0$', r'$\xi^{02}_1$',
    r'$\xi^{12}_0$', r'$\xi^{12}_1$', r'$\tau_e$', r'$\beta$'
    ]
def get_param_dict(plasticity_params):
    return dict(zip(param_names, plasticity_params))


def run_trial_coherence_2afc(
    initialisation_steps=10,
    lambda_=0.8,
    total_time=runtime,
    plasticity=True,
    W=None,
    nu=nu_initial,
    theta=nu_initial,
    coherence=0.5,
    stim_start=0.2,
    stim_end=0.4,
    eval_time=0.4,
    dR_dt=dR_dt_default,
    use_phi_fitted=True,
    plasticity_params=nolearn_parameters,
    randomstate=random_state_default,
):
    """Specifies stimulus and reward function for 2-afc task."""
    assert p >= 2, "2afc requires at least two selective groups"
    if isinstance(coherence, Iterable):
        coherence = randomstate.choice(coherence)

    multiplier = np.ones(p+2)
    multiplier[1] += 0.05 * (1+coherence)
    multiplier[2] += 0.05 * (1-coherence)

    def reward_func(nu):
        reward = -1.
        if np.sign(coherence) == 1:
            if np.argmax(nu[1:-1]) == 0:  # new indexing
                reward = 1.
        elif np.sign(coherence) == -1:
            if np.argmax(nu[1:-1]) == 1:
                reward = 1. 
        elif np.sign(coherence) == 0.:
            reward = np.random.choice([-1,1])
        # scale for fixed total reward
        # still, for summing, one should scale by defaultdt
        return reward / tau_reward_default
    
    if use_phi_fitted:
        return run_trial_fitted(initialisation_steps=initialisation_steps,
            lambda_=lambda_,
            total_time=total_time,
            plasticity=plasticity,
            W=W,
            nu=nu_initial,
            theta=nu_initial,
            external_input_multiplier=multiplier,
            stim_start=stim_start,
            stim_end=stim_end,
            eval_time=eval_time,
            reward_func=reward_func,
            plasticity_params=plasticity_params,
            randomstate=randomstate
        )
    return run_trial(
        initialisation_steps=initialisation_steps,
        lambda_=lambda_,
        total_time=total_time,
        plasticity=plasticity,
        W=W,
        nu=nu_initial,
        theta=nu_initial,
        external_input_multiplier=multiplier,
        stim_start=stim_start,
        stim_end=stim_end,
        eval_time=eval_time,
        reward_func=reward_func,
        dR_dt=dR_dt,
        use_phi_fitted=use_phi_fitted,
        plasticity_params=plasticity_params,
        randomstate=randomstate,
    )


def run_trial_XOR(
    initialisation_steps=10,
    lambda_=0.8,
    total_time=runtime,
    plasticity=True,
    W=None,
    nu=nu_initial,
    theta=nu_initial,
    coherence=0.5,
    stim_start=0.2,
    stim_end=0.4,
    eval_time=0.6,
    dR_dt=dR_dt_default,
    use_phi_fitted=True,
    plasticity_params=nolearn_parameters,
    randomstate=random_state_default,
):
    """Specifies stimulus and reward function for XOR task."""
    assert p >= 4, "XOR needs two inputs and two outputs"

    if isinstance(coherence, Iterable):
        coherence = randomstate.choice(coherence)
    
    multiplier = np.ones(p+2)
    inputs = np.random.choice([0, 1], size=2)
    readout_cell = 2 if inputs.sum() % 2 == 0 else 3
    nonreadout_cell = 3 if inputs.sum() % 2 == 0 else 2
    multiplier[1:3] += 0.05*(1 + inputs*coherence)

    def reward_func(nu):
        reward = -1.
        # harder alternative
        # if np.argmax(nu[1:-1]) == readout_cell:
        #     reward = 1.
        # easier alternative
        if nu[nonreadout_cell] < nu[readout_cell]:
            reward = 1.
        return reward / tau_reward_default
        
    if use_phi_fitted:
        return run_trial_fitted(initialisation_steps=initialisation_steps,
            lambda_=lambda_,
            total_time=total_time,
            plasticity=plasticity,
            W=W,
            nu=nu_initial,
            theta=nu_initial,
            external_input_multiplier=multiplier,
            stim_start=stim_start,
            stim_end=stim_end,
            eval_time=eval_time,
            reward_func=reward_func,
            plasticity_params=plasticity_params,
            randomstate=randomstate
        )
    return run_trial(
        initialisation_steps=initialisation_steps,
        lambda_=lambda_,
        total_time=total_time,
        plasticity=plasticity,
        W=W,
        nu=nu_initial,
        theta=nu_initial,
        external_input_multiplier=multiplier,
        stim_start=stim_start,
        stim_end=stim_end,
        eval_time=eval_time,
        reward_func=reward_func,
        dR_dt=dR_dt,
        use_phi_fitted=use_phi_fitted,
        plasticity_params=plasticity_params,
        randomstate=randomstate,
    )

def run_trial_fitted(
    initialisation_steps=10,
    lambda_=0.8,
    total_time=runtime,
    plasticity=True,
    W=None,
    nu=nu_initial,
    theta=nu_initial,
    external_input_multiplier=np.ones(p+2),
    stim_start=0.2,
    stim_end=0.4,
    eval_time=0.4,
    reward_func=lambda x: 0.0,
    plasticity_params=nolearn_parameters,
    randomstate=random_state_default,
):
    if W is None:
        W = get_weights()
    else:
        W = W.copy()
    W = W.clip(0.0, w_max_default)
    nu = nu.copy()
    theta = theta.copy()

    # AMPA
    s_AMPA = tau_AMPA * nu
    s_AMPA[~pyramidal_mask] = 0.  # inhibitory neurons won't feed AMPA-mediated synapses
    ip_AMPA = (V_drive - V_E) * C_k * s_AMPA
    ic_AMPA = g_AMPA * (W @ ip_AMPA)

    # AMPA_ext
    s_AMPA_ext = np.full_like(
        s_AMPA,
        tau_AMPA * rate_ext
    )  # array to allow for differing inputs
    ip_AMPA_ext = (V_drive - V_E) * C_ext * s_AMPA_ext
    ic_AMPA_ext = g_AMPA_ext * ip_AMPA_ext

    # GABA
    s_GABA = tau_GABA * nu
    s_GABA[pyramidal_mask] = 0.
    ip_GABA = (V_drive - V_I) * C_k * s_GABA
    ic_GABA = g_GABA * (W @  ip_GABA)

    s_NMDA = psi(nu)
    s_NMDA[~pyramidal_mask] = 0.
    
    # Initialise V_avg, V_SS
    V_avg = np.full(p+2, V_avg_initial)
    for k in range(initialisation_steps):
        g_NMDA_eff_V = g_NMDA_eff(V_avg)
        V_E_eff_V = V_E_eff(V_avg)
        ip_NMDA = (V_drive - V_E_eff_V) * C_k * s_NMDA
        ic_NMDA = g_NMDA_eff_V * (W @ ip_NMDA)
        I_syn = ic_AMPA + ic_AMPA_ext + ic_NMDA + ic_GABA
        V_SS = V_L - I_syn / g_m
        V_avg = V_SS - (V_thr-V_reset)*nu*tau_m - (V_SS-V_reset)*nu*tau_rp
    
    sigma = get_sigma(lambda_=lambda_, s_AMPA_ext=s_AMPA_ext)

    ## Initialise arrays to track values, and for simulation
    times = np.arange(0, total_time, defaultdt)
    nu_tracked = np.full((p+2, times.shape[0]), np.nan)
    s_NMDA_tracked = np.full((p+2, times.shape[0]), np.nan)
    s_AMPA_tracked = np.full((p+2, times.shape[0]), np.nan)
    s_GABA_tracked = np.full((p+2, times.shape[0]), np.nan)
    I_syn_tracked = np.full((p+2, times.shape[0]), np.nan)
    theta_tracked = np.full((p+2, times.shape[0]), np.nan)
    reward_tracked = np.full((1, times.shape[0]), np.nan)  # only one reward signal
    e_tracked = np.full((p+2, p+2, times.shape[0]), np.nan)
    W_tracked = np.full((p+2, p+2, times.shape[0]), np.nan)

    if not plasticity:
        W_tracked = np.full((p+2, p+2, times.shape[0]), W.reshape(p+2, p+2, 1))
    
    if theta is None:
        theta = nu
    ic_noise = np.zeros_like(s_AMPA)
    e = np.zeros_like(W)
    reward = 0.
    has_evaluated = False
    stimulated_bool = False
    for itr, t in enumerate(times):
        if t > stim_start and t < stim_end and not stimulated_bool:
            s_AMPA_ext = s_AMPA_ext * external_input_multiplier
            stimulated_bool = True
        elif t >= stim_end and stimulated_bool:
            s_AMPA_ext = np.full_like(
                s_AMPA,
                tau_AMPA * rate_ext
            )
            stimulated_bool = False
        I_syn, nu, s_NMDA, s_AMPA, \
        s_GABA, ic_noise, reward, V_avg, \
        theta, e, W = compute_update_step(
            sigma=sigma, V_avg=V_avg, nu=nu,
            s_AMPA_ext=s_AMPA_ext, s_AMPA=s_AMPA,
            s_NMDA=s_NMDA, s_GABA=s_GABA, ic_noise=ic_noise,
            reward=reward, W=W, theta=theta, e=e,
            randomstate=randomstate, plasticity=plasticity,
            plasticity_params=plasticity_params
        )
        if np.any(np.isnan(nu)) or np.any(np.isnan(W)):
            # break before storing the NaNs
            break
        if not has_evaluated and t >= eval_time:
            has_evaluated = True
            reward = reward_func(nu)
        
        nu_tracked[:, itr] = nu
        s_NMDA_tracked[:, itr] = s_NMDA
        s_AMPA_tracked[:, itr] = s_AMPA
        s_GABA_tracked[:, itr] = s_GABA
        I_syn_tracked[:, itr] = I_syn
        reward_tracked[:, itr] = reward
        
        if plasticity:
            theta_tracked[:, itr] = theta
            e_tracked[:, :, itr] = e
            W_tracked[:, :, itr] = W
        
    
    return_dict = dict(
        times=times,
        nu=nu_tracked,
        s_NMDA=s_NMDA_tracked,
        s_AMPA=s_AMPA_tracked,
        s_GABA=s_GABA_tracked,
        I_syn=I_syn_tracked,
        theta=theta_tracked,
        e=e_tracked,
        W=W_tracked,
        reward=reward_tracked
    )
    return return_dict

# @jit  # faster to not plain jit
def run_trial(
    initialisation_steps=10,
    lambda_=0.8,
    total_time=runtime,
    plasticity=True,
    W=None,
    nu=nu_initial,
    theta=nu_initial,
    external_input_multiplier=np.ones(p+2),
    stim_start=0.2,
    stim_end=0.4,
    eval_time=0.4,
    reward_func=lambda x: 0.0,  # TODO
    dR_dt=dR_dt_default,
    use_phi_fitted=True,
    plasticity_params=nolearn_parameters,
    randomstate=random_state_default,
):  
    # set weights
    # TODO: refactor so that W is input,
    # and asserts are not needed
    nu = nu.copy()
    theta = theta.copy()
    if W is None:
        W = get_weights()
    else:
        assert W.shape[0] == p+2 
        W = W.copy()
    assert len(external_input_multiplier) == p+2
    W = W.clip(0.0, w_max_default)
    # AMPA
    s_AMPA = tau_AMPA * nu
    s_AMPA[~pyramidal_mask] = 0.  # inhibitory neurons won't feed AMPA-mediated synapses
    ip_AMPA = (V_drive - V_E) * C_k * s_AMPA
    ic_AMPA = g_AMPA * (W @ ip_AMPA)

    # AMPA_ext
    s_AMPA_ext = np.full_like(
        s_AMPA,
        tau_AMPA * rate_ext
    )  # array to allow for differing inputs
    ip_AMPA_ext = (V_drive - V_E) * C_ext * s_AMPA_ext
    ic_AMPA_ext = g_AMPA_ext * ip_AMPA_ext

    # GABA
    s_GABA = tau_GABA * nu
    s_GABA[pyramidal_mask] = 0.
    ip_GABA = (V_drive - V_I) * C_k * s_GABA
    ic_GABA = g_GABA * (W @  ip_GABA)

    s_NMDA = psi(nu)
    s_NMDA[~pyramidal_mask] = 0.
    
    # Initialise V_avg, V_SS
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
    # sigma = np.sqrt(
    #         g_AMPA_ext**2 * (V_drive - V_E)**2 * C_ext * s_AMPA_ext * tau_AMPA / (g_m**2 * tau_m)
    #     )
    # sigma[:-1] = lambda_ * (2e-3) + (1-lambda_) * sigma[:-1]
    sigma = get_sigma(lambda_=lambda_, s_AMPA_ext=s_AMPA_ext)
    
    ## Initialise arrays to track values, and for simulation
    times = np.arange(0, total_time, defaultdt)
    nu_tracked = np.full((p+2, times.shape[0]), np.nan)
    s_NMDA_tracked = np.full((p+2, times.shape[0]), np.nan)
    s_AMPA_tracked = np.full((p+2, times.shape[0]), np.nan)
    s_GABA_tracked = np.full((p+2, times.shape[0]), np.nan)
    I_syn_tracked = np.full((p+2, times.shape[0]), np.nan)
    theta_tracked = np.full((p+2, times.shape[0]), np.nan)
    reward_tracked = np.full((1, times.shape[0]), np.nan)  # only one reward signal
    e_tracked = np.full((p+2, p+2, times.shape[0]), np.nan)
    W_tracked = np.full((p+2, p+2, times.shape[0]), np.nan)
    
    if not plasticity:
        W_tracked = np.full((p+2, p+2, times.shape[0]), W.reshape(p+2, p+2, 1))
    
    ic_noise = np.zeros_like(s_AMPA)
    e = np.zeros_like(W)
    reward = 0.
    has_evaluated = False
    for itr, t in enumerate(times):
        #TODO refactor so that this is the update step
        # one for phi_fitted and one for the original
        # compute reward with eval funcoutside of update,
        # after update
        # also update s_AMPA_ext outside of this loop
        
        ip_AMPA = (V_drive - V_E) * C_k * s_AMPA
        ic_AMPA = g_AMPA * (W @ ip_AMPA)

        g_NMDA_eff_V = g_NMDA_eff(V_avg)
        V_E_eff_V = V_E_eff(V_avg)
        ip_NMDA = (V_drive - V_E_eff_V) * C_k * s_NMDA
        ic_NMDA = g_NMDA_eff_V * (W @ ip_NMDA)

        ip_GABA = (V_drive - V_I) * C_k * s_GABA
        ic_GABA = g_GABA * (W @  ip_GABA)
        
        # TODO: don't recompute on each step
        s_AMPA_ext = np.full_like(
            s_AMPA,
            tau_AMPA * rate_ext
        )
        if t > stim_start and t < stim_end:
            s_AMPA_ext = s_AMPA_ext * external_input_multiplier
            
        ip_AMPA_ext = (V_drive - V_E) * C_ext * s_AMPA_ext
        ic_AMPA_ext = g_AMPA_ext * ip_AMPA_ext

        # TODO: compute outside of loop, change only when needed
        sigma = get_sigma(lambda_=lambda_, s_AMPA_ext=s_AMPA_ext)
        # sigma = np.sqrt(
        #     g_AMPA_ext**2 * (V_drive - V_E)**2 * C_ext * s_AMPA_ext * tau_AMPA / (g_m**2 * tau_m)
        # )
        # sigma[:-1] = lambda_ * (2e-3) + (1-lambda_) * sigma[:-1]
        
        I_syn = ic_AMPA + ic_AMPA_ext + ic_NMDA + ic_GABA + ic_noise
        if use_phi_fitted:
            dnu_dt_now = dnu_dt_fitted(
                nu, I_syn,
                sigma=sigma,
                g_m=g_m,
                tau_m=tau_m,
                tau_rp=tau_rp
            )
        else:
            dnu_dt_now = dnu_dt(
                nu, I_syn,
                sigma=sigma,
                g_m=g_m,
                tau_m=tau_m,
                tau_rp=tau_rp
            )
        dic_noise_dt_now = dic_noise_dt(
            ic_noise, randomstate=randomstate
            )
        ds_NMDA_dt_now = ds_NMDA_dt(s_NMDA, nu)
        ds_AMPA_dt_now = ds_AMPA_dt(s_AMPA, nu)
        ds_GABA_dt_now = ds_GABA_dt(s_GABA, nu)

        if not has_evaluated and t >= eval_time:
            has_evaluated = True
            reward = reward_func(nu)
            dreward_dt_now = 0.  # really the derivative is a Dirac "function"
        else:
            dreward_dt_now = dR_dt(reward)
        
        if plasticity:
            dtheta_dt_now = dtheta_dt(
                theta=theta, nu=nu,
                plasticity_params=plasticity_params
            )
            F_val = F_full(
                nu=nu, W=W, theta=theta,
                plasticity_params=plasticity_params
            )
            de_dt_now = de_dt(
                W=W, eligibility_trace=e, nu=nu,
                theta=theta, F_val=F_val,
                plasticity_params=plasticity_params
            )
            dW_dt_now = dW_dt(
                W=W, eligibility_trace=e, nu=nu,
                reward=reward, theta=theta, F_val=F_val,
                plasticity_params=plasticity_params
            )
            
        nu += dnu_dt_now * defaultdt
        ic_noise += dic_noise_dt_now * defaultdt
        s_NMDA += ds_NMDA_dt_now * defaultdt
        s_AMPA += ds_AMPA_dt_now * defaultdt
        s_GABA += ds_GABA_dt_now * defaultdt
        reward += dreward_dt_now * defaultdt
        
        if plasticity:
            theta += dtheta_dt_now * defaultdt
            e += de_dt_now * defaultdt
            W += dW_dt_now * defaultdt
            W = W.clip(0.0, w_max_default) # upper bound chosen for stability
            
        if np.any(np.isnan(nu)) or np.any(np.isnan(W)):
            # break before storing the NaNs
            break

        # TODO: determine reward here, at end of loop, so at the start
        # of the next timestep the plasticity and dynamics get full reward
            
        nu_tracked[:, itr] = nu
        s_NMDA_tracked[:, itr] = s_NMDA
        s_AMPA_tracked[:, itr] = s_AMPA
        s_GABA_tracked[:, itr] = s_GABA
        I_syn_tracked[:, itr] = I_syn
        reward_tracked[:, itr] = reward
        
        if plasticity:
            theta_tracked[:, itr] = theta
            e_tracked[:, :, itr] = e
            W_tracked[:, :, itr] = W
            
        # Not mentioned in W&W2006:
        # update V_avg for NMDA channel effects
        V_SS = V_L - I_syn / g_m
        V_avg = V_SS - (V_thr-V_reset)*nu*tau_m - (V_SS-V_reset)*nu*tau_rp

    
    return_dict = dict(
        times=times,
        nu=nu_tracked,
        s_NMDA=s_NMDA_tracked,
        s_AMPA=s_AMPA_tracked,
        s_GABA=s_GABA_tracked,
        I_syn=I_syn_tracked,
        theta=theta_tracked,
        e=e_tracked,
        W=W_tracked,
        reward=reward_tracked
    )
    return return_dict

if __name__ == '__main__':
    W = get_weights(w_plus=3.1, w_minus=0.1)
    results = run_trial_coherence_2afc(W=W, initialisation_steps=100)
    print(results)
