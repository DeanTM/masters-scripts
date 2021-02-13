from parameters import *

import numba
from numba import jit

# import numpy as np
from scipy import special, integrate

import warnings
warnings.filterwarnings('ignore')

# pyramidal mask used to determine which vector values to zero
pyramidal_mask = np.array([True] * (p+1) + [False])
# plasticity masks used in determining which weights to update
plasticity_mask_source = pyramidal_mask.copy()
plasticity_mask_target = np.full_like(plasticity_mask_source, True)

# sigma
@jit(nopython=True)
def get_sigma(lambda_=0.8):
    sigma = np.sqrt(
        g_AMPA_ext**2 * (V_drive - V_E)**2 * C_ext * rate_ext * tau_AMPA**2 / (g_m**2 * tau_m)
    )
    sigma[:-1] = lambda_ * (2e-3) + (1-lambda_) * sigma[:-1]
    return sigma

# Jahr-Stevens formula functions
@jit(nopython=True)
def J(V):
    result = np.empty_like(V)
    for i in range(V.shape[0]):
        result[i] = 1 + gamma_JahrStevens * np.exp(-beta_JahrStevens * V[i])
    return result


@jit(nopython=True)
def J_2(V):
    result = np.empty_like(V)
    J_V = J(V)
    for i in range(V.shape[0]):
        numerator = J_V[i] + beta_JahrStevens * (V[i] - V_E)*(J_V[i] - 1)
        denominator = J_V[i]**2
        result[i] = numerator / denominator
    return result

@jit(nopython=True)
def g_NMDA_eff(V):
    return g_NMDA * J_2(V)
    
@jit(nopython=True)
def V_E_eff(V):
    return V - (1 / J_2(V)) * (V - V_E) / J(V)

#region Functions to compute steady-state NMDA channels
@jit(nopython=True)
def my_factorial(n):
    if n <= 1.:
        return 1.
    return my_factorial(n-1) * n

@jit(nopython=True)
def my_binomial(n, m):
    return my_factorial(n) / (my_factorial(m) * my_factorial(n-m))


@jit(nopython=True)
def _get_Tn_summand(n, m, nu):
    assert n >= 1 and m >= 0
    binom_coeff = (-1)**m * my_binomial(n, m)
    result = np.empty_like(nu)
    for i in range(nu.shape[0]):
        numerator = tau_NMDA_rise * (1 + nu[i] * tau_NMDA)
        denominator = tau_NMDA_rise * (1 + nu[i] * tau_NMDA) + m * tau_NMDA_decay
        result[i] = binom_coeff * numerator / denominator
    return result

def _get_Tn_summand_scipy(n, m, nu):
    assert n >= 1 and m >= 0
    binom_coeff = (-1)**m * special.binom(n, m)
    numerator = tau_NMDA_rise * (1 + nu * tau_NMDA)
    denominator = tau_NMDA_rise * (1 + nu * tau_NMDA) + m * tau_NMDA_decay
    return binom_coeff * numerator / denominator

@jit(nopython=True)
def _get_Tn(n, nu):
    assert n >= 1
    Tn = np.zeros_like(nu)
    for m in np.arange(0, n+1):
        Tn += _get_Tn_summand(n, m, nu)
    return Tn

def _get_Tn_scipy(n, nu):
    assert n >= 1
    Tn = 0
    for m in np.arange(0, n+1):
        Tn += _get_Tn_summand_scipy(n, m, nu)
    return Tn

@jit(nopython=True)
def psi(nu, n_truncate=5):
    """
    Computes fraction of open NMDA channels given presynaptic firing rate nu 
    up until truncation round-off n_truncate.
    """
    coeff = nu * tau_NMDA / (1 + nu * tau_NMDA)
    summation = np.zeros_like(nu)
    for n in np.arange(1, n_truncate+1):
        Tn = _get_Tn(n, nu)
        summand_coeff = ((-alpha * tau_NMDA_rise)**n) / my_factorial(n+1)
        summation += summand_coeff * Tn
    return coeff * (1 + summation / (1 + nu * tau_NMDA))

def psi_scipy(nu, n_truncate=5):
    """
    Computes fraction of open NMDA channels given presynaptic firing rate nu 
    up until truncation round-off n_truncate.
    """
    coeff = nu * tau_NMDA / (1 + nu * tau_NMDA)
    summation = 0
    for n in np.arange(1, n_truncate+1):
        summand_coeff = ((-alpha * tau_NMDA_rise)**n) / special.factorial(n+1)
        summation += summand_coeff * _get_Tn_scipy(n, nu)
    return coeff * (1 + summation / (1 + nu * tau_NMDA))
#endregion

#region Firing rate functions
@jit(nopython=True)
def _rate_upperbound_vectorised(
    V_SS, sigma, tau_m, tau_rp,
):
    summand = (V_thr - V_SS)/sigma
    summand *= 1 + 0.5 * tau_AMPA/tau_m
    summand += 1.03 * np.sqrt(tau_AMPA/tau_m) - 0.5 * tau_AMPA/tau_m
    return summand

@jit(nopython=True)
def _rate_lowerbound_vectorised(
    V_SS, sigma,
):
    return (V_reset - V_SS) / sigma

def _siegert_integrand(x):
    return np.exp(x**2)*(1+special.erf(x))


root_pi = np.sqrt(np.pi)
def rate_vectorised(
    V_SS, sigma, tau_m, tau_rp
):
    integration_results = np.empty(V_SS.shape)
    UB = _rate_upperbound_vectorised(
        V_SS, sigma,
        tau_m=tau_m, tau_rp=tau_rp
    )
    LB = _rate_lowerbound_vectorised(
        V_SS, sigma
    )
    # There would be a dynamic programming solution
    # to computing multiple integrals by splitting 
    # this up into subintervals, integrating over 
    # them and adding them back together
    for i, (lb, ub) in enumerate(zip(LB, UB)):
        integral, error = integrate.quad(
            _siegert_integrand, lb, ub, limit=200
        )
        integration_results[i] = integral
    return (tau_rp + tau_m * root_pi * integration_results)**-1

def phi(
    I_syn, g_m, sigma, tau_m, tau_rp
):
    """
    Firing rate function written to take input current for simplicity.
    """
    V_SS = V_L - I_syn/g_m
    return rate_vectorised(V_SS, sigma, tau_m, tau_rp)


c_default = 310.*1e9
I_default = 125.
g_default = 0.16
@jit(nopython=True)
def phi_fit(
    I_syn,
    c=c_default,
    I=I_default,
    g=g_default
):
    numerator = c * (-I_syn) - I
    denominator = 1 - np.exp(-g*(numerator))
    return numerator / denominator


@jit(nopython=True)
def polyfunc(x, coeff):
    s = np.zeros_like(x)
    for n, c_ in enumerate(coeff[::-1]):
        s += c_ * (x**n)
    return s


polyfit_coeffs_E = np.load('polyfit_coeffs_excitatory.npy')
max_rate_E = 1./tau_rp_E
@jit(nopython=True)
def phi_fit_E(I_syn, sigma):
    c = polyfunc(sigma, coeff=polyfit_coeffs_E[0, :])
    I = polyfunc(sigma, coeff=polyfit_coeffs_E[1, :])
    g = polyfunc(sigma, coeff=polyfit_coeffs_E[2, :])
    rates = phi_fit(I_syn, c,I,g)
    rates[rates > max_rate_E] = max_rate_E
    return rates

# sigma is ignored parameter, kept for signatures to match
c_I, I_I, g_I = np.load('direct_fit_inhibitory.npy')
max_rate_I = 1./tau_rp_I
@jit(nopython=True)
def phi_fit_I(I_syn, sigma=None):
    rates = phi_fit(I_syn, c_I, I_I, g_I)
    rates[rates > max_rate_I] = max_rate_I
    return rates
#endregion

#region Euler derivative updates

#region activity variables
# @jit(nopython=True)  # faster without jit for now
def ds_NMDA_dt(s_NMDA, nu):
    psi_nu = psi(nu)
    psi_nu[~pyramidal_mask] = 0.0
    tau_NMDA_eff = tau_NMDA * (1 - psi_nu)
    dsdt = -(s_NMDA - psi_nu) / tau_NMDA_eff
    return dsdt

# @jit(nopython=True)
def dic_noise_dt(
    ic_noise,
    sigma_noise=7e-12, # should this scale with number of populations?
    randomstate=random_state_default
):
    # eta = np.random.randn(*ic_noise.shape)
    eta = randomstate.randn(*ic_noise.shape)
    dicdt = (-ic_noise + eta * np.sqrt(tau_AMPA/defaultdt) * sigma_noise) / tau_AMPA
    return dicdt

def dnu_dt(
    nu, I_syn, g_m, sigma, tau_m, tau_rp
):
    phi_Isyn = phi(
        I_syn, g_m, sigma, tau_m, tau_rp
    )
    deriv = (-nu + phi_Isyn)/tau_rate
    return deriv

def dnu_dt_fitted(
    nu, I_syn, g_m, sigma, tau_m, tau_rp
):
    phi_Isyn = np.zeros_like(nu)
    phi_Isyn[pyramidal_mask] = phi_fit_E(
        I_syn[pyramidal_mask], sigma[pyramidal_mask],
    )
    phi_Isyn[~pyramidal_mask] = phi_fit_I(
        I_syn[~pyramidal_mask], sigma[~pyramidal_mask],
    )
    deriv = (-nu + phi_Isyn)/tau_rate
    return deriv

@jit(nopython=True)
def ds_AMPA_dt(s_AMPA, nu):
    deriv = -s_AMPA/tau_AMPA + nu
    deriv[~pyramidal_mask] = 0.0
    return deriv

@jit(nopython=True)
def ds_GABA_dt(s_GABA, nu):
    deriv = -s_GABA/tau_GABA + nu
    deriv[pyramidal_mask] = 0.0
    return deriv
#endregion
    

#region plasticity variables
# TODO: test
@jit(nopython=True)
def F_full(nu, W, theta, plasticity_params):
    # crudely coded
    p_const, p_theta, mu, tau_theta, xi_00, \
    xi_10_0, xi_10_1, xi_01_0, xi_01_1, xi_11_0, xi_11_1, \
    xi_20_0, xi_20_1, xi_21_0, xi_21_1, xi_02_0, xi_02_1, \
    xi_12_0, xi_12_1, tau_e, beta = plasticity_params
    
    nu = nu.reshape(-1,1)  # make sure it's a column
    theta = theta.reshape(-1, 1)
    theta_cast = theta @ np.ones_like(theta).T
    theta_cast_p = theta_cast**p_theta
    ones_vec = np.ones_like(nu)
    result = np.zeros_like(W)
        
    result += (xi_10_0 + xi_10_1 * theta_cast_p) * (nu @ ones_vec.T)
    result += (xi_01_0 + xi_01_1 * theta_cast_p) * (ones_vec @ nu.T)
    result += (xi_11_0 + xi_11_1 * theta_cast_p) * (nu @ nu.T)
    result += (xi_20_0 + xi_20_1 * theta_cast_p) * ((nu**2) @ ones_vec.T)
    result += (xi_21_0 + xi_21_1 * theta_cast_p) * ((nu**2) @ nu.T)
    result += (xi_02_0 + xi_02_1 * theta_cast_p) * (ones_vec @ (nu.T**2))
    result += (xi_12_0 + xi_12_1 * theta_cast_p) * (nu @ (nu.T**2))
    result *= ((w_max_default - W)*W)**mu
    result += xi_00 * (theta_cast**p_const) * W
    return result

## not used:
# tau_theta = 0.1
# @jit(nopython=True)
# def dtheta_BCM_dt(theta, nu):
#     """
#     tau_theta * dtheta/dt = -theta + nu**2
    
#     Units of theta are technically different to those of nu. 
#     There should be a constant to fix this.
#     """
#     dtheta_dt = (-theta + nu**2)/tau_theta
#     return dtheta_dt

# @jit(nopython=True)
def dtheta_dt(
    theta, nu,
    plasticity_params,
    **kwargs
):
    """
    Threshold for the BCM-rule.
    
    tau_theta * dtheta/dt = -theta + nu**2
    
    Units of theta are technically different to those of nu. 
    There should be a constant to fix this.
    """
    if theta is None:
        return None
    p_const, p_theta, mu, tau_theta, xi_00, \
    xi_10_0, xi_10_1, xi_01_0, xi_01_1, xi_11_0, xi_11_1, \
    xi_20_0, xi_20_1, xi_21_0, xi_21_1, xi_02_0, xi_02_1, \
    xi_12_0, xi_12_1, tau_e, beta = plasticity_params
    dtheta_dt_now = (-theta + nu)/tau_theta
    return dtheta_dt_now

# TODO: refactor so that F_val is given
# @jit(nopython=True)
def de_dt(
    W, eligibility_trace, nu,
    plasticity_params,
    theta=None,
    F_val=None,
    **kwargs
):
    p_const, p_theta, mu, tau_theta, xi_00, \
    xi_10_0, xi_10_1, xi_01_0, xi_01_1, xi_11_0, xi_11_1, \
    xi_20_0, xi_20_1, xi_21_0, xi_21_1, xi_02_0, xi_02_1, \
    xi_12_0, xi_12_1, tau_e, beta = plasticity_params
    if F_val is None:
        F_val = F_full(nu, W, theta, plasticity_params)
    tau_de_dt = -eligibility_trace + F_val
    return tau_de_dt / tau_e


# TODO: refactor so that F_val is given
# @jit(nopython=True)
def dW_dt(
    W, eligibility_trace, nu, reward,
    plasticity_params,
    theta=None,
    F_val=None,
    **kwargs
):
    p_const, p_theta, mu, tau_theta, xi_00, \
    xi_10_0, xi_10_1, xi_01_0, xi_01_1, xi_11_0, xi_11_1, \
    xi_20_0, xi_20_1, xi_21_0, xi_21_1, xi_02_0, xi_02_1, \
    xi_12_0, xi_12_1, tau_e, beta = plasticity_params
    if F_val is None:
        if beta > 0.:
            F_val = F_full(nu, W, theta, plasticity_params)
        else:
            F_val = 0.0
    tau_dw_dt = (1-beta)*eligibility_trace*reward + beta*F_val
    return tau_dw_dt  # / tau_w  # tau_w is redundant

#endregion

@jit(nopython=True)
def dR_dt_default(reward, tau_reward=tau_reward_default):
    dR_dt_now = -reward/tau_reward
    return dR_dt_now


#endregion

@jit(nopython=True)
def dW_dt_BCM(W, nu, theta):
    dW_dt = (np.outer(nu * (nu-theta), nu) / theta.reshape(-1, 1)) / tau_W
    for j, b in enumerate(plasticity_mask_source):
        if not b:
            dW_dt[:, j] = 0.0
    for i, b in enumerate(plasticity_mask_target):
        if not b:
            dW_dt[i, :] = 0.0
    return dW_dt

@jit(nopython=True)
def get_w_minus(w_plus=w_plus, f=f):
    return 1.0 - f*(w_plus - 1.0) / (1.0 - f)

@jit(nopython=True)
def get_weights(w_plus=w_plus, p=p, f=f, w_minus=w_minus):
    W = np.ones((p+2, p+2))  # from column to row
    for i in range(0, p+1):
        weights = np.full(p+2, w_minus)
        if i > 0:
            weights[i] = w_plus
        else:
            weights[i] = 1.0
        weights[-1] = 1.0
        weights[0] = 1.0
        W[:,i] = weights
    return W

# TODO: remove this, it's obsolete
def simulate_original(
    initialisation_steps=10,
    lambda_=0.8,
    total_time=runtime,
    coherence=0.5,
    plasticity=True,
    W=None,
    show_time=False,
    use_phi_fitted=False
):
    """
    The original simulation for a n-AFC task with one external stimulus.
    Used for debugging.
    
    Added feature for testing: to use the fitted phi func.
    """
    ## Initialise arrays for computation
    # For some reason this can't be numba.jit-ed
    # C_k = np.array([N_non] + [N_sub] * p + [N_I])
    # g_m = np.array([g_m_E] * (p+1) + [g_m_I])
    # C_m = np.array([C_m_E] * (p+1) + [C_m_I])
    # tau_m = np.array([tau_m_E] * (p+1) + [tau_m_I])
    # tau_rp = np.array([tau_rp_E] * (p+1) + [tau_rp_I])
    # nu = np.array([rate_pyramidal] * (p+1) + [rate_interneuron])
    nu = nu_initial
    
    # set weights
    if W is None:
        W = get_weights()
    else:
        assert W.shape[0] == p+2 
        W = W.copy()

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
    V_avg_initial = -56e-3  # initial guess for determining V_avg
    
    # Initialise V_avg, V_SS
    # can I speed this up with @numba.jit?
    V_avg = np.full(p+2, V_avg_initial)
    for k in range(initialisation_steps):
        g_NMDA_eff_V = g_NMDA_eff(V_avg)
        V_E_eff_V = V_E_eff(V_avg)
        ip_NMDA = (V_drive - V_E_eff_V) * C_k * s_NMDA
        ic_NMDA = g_NMDA_eff_V * (W @ ip_NMDA)
        I_syn = ic_AMPA + ic_AMPA_ext + ic_NMDA + ic_GABA
        V_SS = V_L - I_syn / g_m  # notice the minus, because current flows out?
        V_avg = V_SS - (V_thr-V_reset)*nu*tau_m - (V_SS-V_reset)*nu*tau_rp
    
    # set sigma
    sigma = np.sqrt(
        g_AMPA_ext**2 * (V_drive - V_E)**2 * C_ext * rate_ext * tau_AMPA**2 / (g_m**2 * tau_m)
    )
    sigma[:-1] = lambda_ * (2e-3) + (1-lambda_) * sigma[:-1]

    # initialise values for learning
    theta_BCM = nu ** 2

    ## Initialise arrays to track values, and for simulation
    times = np.arange(0, total_time, defaultdt)
#     nu_tracked = np.zeros((p+2, times.shape[0]))
#     s_NMDA_tracked = np.zeros((p+2, times.shape[0]))
#     s_AMPA_tracked = np.zeros((p+2, times.shape[0]))
#     s_GABA_tracked = np.zeros((p+2, times.shape[0]))
#     I_syn_tracked = np.zeros((p+2, times.shape[0]))
    
#     theta_BCM_tracked = np.zeros((p+2, times.shape[0]))
#     W_tracked = np.zeros((p+2, p+2, times.shape[0]))
    nu_tracked = np.full((p+2, times.shape[0]), np.nan)
    s_NMDA_tracked = np.full((p+2, times.shape[0]), np.nan)
    s_AMPA_tracked = np.full((p+2, times.shape[0]), np.nan)
    s_GABA_tracked = np.full((p+2, times.shape[0]), np.nan)
    I_syn_tracked = np.full((p+2, times.shape[0]), np.nan)
    
    theta_BCM_tracked = np.full((p+2, times.shape[0]), np.nan)
    W_tracked = np.full((p+2, p+2, times.shape[0]), np.nan)
    
    if not plasticity:
        W_tracked = np.full((p+2, p+2, times.shape[0]), W.reshape(p+2, p+2, 1))
    
    ic_noise = np.zeros_like(s_AMPA)
    if show_time:
        from tqdm import tqdm
    else:
        tqdm = lambda x: x
    for itr, t in tqdm(enumerate(times)):
        ip_AMPA = (V_drive - V_E) * C_k * s_AMPA
        ic_AMPA = g_AMPA * (W @ ip_AMPA)

        g_NMDA_eff_V = g_NMDA_eff(V_avg)
        V_E_eff_V = V_E_eff(V_avg)
        ip_NMDA = (V_drive - V_E_eff_V) * C_k * s_NMDA
        ic_NMDA = g_NMDA_eff_V * (W @ ip_NMDA)

        ip_GABA = (V_drive - V_I) * C_k * s_GABA
        ic_GABA = g_GABA * (W @  ip_GABA)


        ## LATEST CHANGE: added noise adapting to changing input by adapting s_AMPA_ext
        s_AMPA_ext = np.full_like(
            s_AMPA,
            tau_AMPA * rate_ext
        )
        if t > 200e-3 and t < 400e-3:
            # mu_0 ~= 0.05 in Wong&Wang2006, taken by comparison with mean external input
            multiplier = np.ones(p+2)
            multiplier[1] += 0.05 * (1+coherence)
            multiplier[2] += 0.05 * (1-coherence)
            s_AMPA_ext = s_AMPA_ext * multiplier

            # increase input by roughly 1.5 times
#             s_AMPA_ext[1] = s_AMPA_ext[1] * 1.5  #(25. * b2.Hz + rate_ext) / rate_ext
        
        ip_AMPA_ext = (V_drive - V_E) * C_ext * s_AMPA_ext
        ic_AMPA_ext = g_AMPA_ext * ip_AMPA_ext
        sigma = np.sqrt(
            g_AMPA_ext**2 * (V_drive - V_E)**2 * C_ext * s_AMPA_ext * tau_AMPA / (g_m**2 * tau_m)
        )
        sigma[:-1] = lambda_ * (2e-3) + (1-lambda_) * sigma[:-1]
        ## END OF LAST CHANGE

        I_syn = ic_AMPA + ic_AMPA_ext + ic_NMDA + ic_GABA + ic_noise

        dnu_dt_now = None
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
        dic_noise_dt_now = dic_noise_dt(ic_noise)
        ds_NMDA_dt_now = ds_NMDA_dt(s_NMDA, nu)
        ds_AMPA_dt_now = ds_AMPA_dt(s_AMPA, nu)
        ds_GABA_dt_now = ds_GABA_dt(s_GABA, nu)
        
        if plasticity:
            dtheta_BCM_dt_now = dtheta_BCM_dt(theta_BCM, nu)
            dW_dt_now = dW_dt_BCM(W, nu, theta_BCM)
        

        nu += dnu_dt_now * defaultdt
        ic_noise += dic_noise_dt_now * defaultdt
        s_NMDA += ds_NMDA_dt_now * defaultdt
        s_AMPA += ds_AMPA_dt_now * defaultdt
        s_GABA += ds_GABA_dt_now * defaultdt
        
        if plasticity:
            theta_BCM += dtheta_BCM_dt_now * defaultdt
            W += dW_dt_now * defaultdt
            W = W.clip(0.0, np.inf)

        nu_tracked[:, itr] = nu
        s_NMDA_tracked[:, itr] = s_NMDA
        s_AMPA_tracked[:, itr] = s_AMPA
        s_GABA_tracked[:, itr] = s_GABA
        I_syn_tracked[:, itr] = I_syn
        
        if plasticity:
            theta_BCM_tracked[:, itr] = theta_BCM
            W_tracked[:, :, itr] = W

        # Not mentioned in W&W2006:
        V_SS = V_L - I_syn / g_m
        V_avg = V_SS - (V_thr-V_reset)*nu*tau_m - (V_SS-V_reset)*nu*tau_rp
        
        if np.any(np.isnan(nu)):
            break

    return  times,\
            nu_tracked,\
            s_NMDA_tracked,\
            s_AMPA_tracked,\
            s_GABA_tracked,\
            I_syn_tracked,\
            theta_BCM_tracked,\
            W_tracked





###############################################
#########  TESTING ############################
###############################################

def test_funcs():
    J_2(np.full(p, V_drive))    
    
    test_rate = np.full(p, rate_ext)
    np.allclose(psi(test_rate), psi_scipy(test_rate))
    
    rate_vectorised(
        np.full(5, V_drive),
        np.full(5, 2e-3),
        tau_m_E, tau_rp_E
    )
    print("Running test simulation")
    times,\
    nu_tracked,\
    s_NMDA_tracked,\
    s_AMPA_tracked,\
    s_GABA_tracked,\
    I_syn_tracked,\
    theta_BCM_tracked,\
    W_tracked = simulate_original(
        total_time=0.6, coherence=0.7,
        plasticity=False,
        show_time=True
    )
    print(f"Simulation ran until {times[-1]:.2f} seconds")
    print("\nRunning fitted-phi simulation")
    times,\
    nu_tracked,\
    s_NMDA_tracked,\
    s_AMPA_tracked,\
    s_GABA_tracked,\
    I_syn_tracked,\
    theta_BCM_tracked,\
    W_tracked = simulate_original(
        total_time=0.6, coherence=0.7,
        plasticity=False,
        show_time=True,
        use_phi_fitted=True
    )
    print(f"Simulation ran until {times[-1]:.2f} seconds")
    
if __name__ == '__main__':
    test_funcs()
    print('Hello World!')