from parameters import *

import numba
from numba import jit

from scipy import special, integrate




# pyramidal mask used to determine which vector values to zero
pyramidal_mask = np.array([True] * (p+1) + [False])
# plasticity masks used in determining which weights to update
plasticity_mask_source = pyramidal_mask.copy()
plasticity_mask_target = np.full_like(plasticity_mask_source, True)
plasticity_mask = np.outer(plasticity_mask_target, plasticity_mask_source)

# sigma
sigma_noAMPA = np.sqrt(
    g_AMPA_ext**2 * (V_drive - V_E)**2 * C_ext * tau_AMPA / (g_m**2 * tau_m)
    )
@jit(nopython=True)
def get_sigma(lambda_=0.8, s_AMPA_ext=rate_ext*tau_AMPA):
    # sigma = np.sqrt(
    #     g_AMPA_ext**2 * (V_drive - V_E)**2 * C_ext * rate_ext * tau_AMPA**2 / (g_m**2 * tau_m)
    # )
    sigma = sigma_noAMPA * np.sqrt(s_AMPA_ext)
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


#FIXME: 1/J_2(V) causes occassional NaNs
@jit(nopython=True)
def V_E_eff(V):
    return V - (1 / J_2(V)) * (V - V_E) / J(V)

def test(v=beta_JahrStevens):
    print(v)

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
@jit(nopython=True)
def ds_NMDA_dt(s_NMDA, nu):
    psi_nu = psi(nu)
    psi_nu[~pyramidal_mask] = 0.0
    tau_NMDA_eff = tau_NMDA * (1 - psi_nu)
    dsdt = -(s_NMDA - psi_nu) / tau_NMDA_eff
    return dsdt

# noise_time = np.sqrt(tau_AMPA/defaultdt) / tau_AMPA
noise_time = 1. / np.sqrt(tau_AMPA * defaultdt)
# @jit(nopython=True)
def dic_noise_dt(
    ic_noise,
    sigma_noise=sigma_noise, #7e-12, # should this scale with number of populations?
    randomstate=random_state_default
):
    eta = randomstate.randn(*ic_noise.shape)
    dicdt = (-ic_noise + eta * sigma_noise) * noise_time
    return dicdt

@jit(nopython=True)
def dic_noise_dt_inputnoise(
    ic_noise,
    eta
):
    dicdt = (-ic_noise + eta * sigma_noise) * noise_time
    return dicdt

def dnu_dt(
    nu, I_syn, g_m, sigma, tau_m, tau_rp
):
    phi_Isyn = phi(
        I_syn, g_m, sigma, tau_m, tau_rp
    )
    deriv = (-nu + phi_Isyn)/tau_rate
    return deriv

@jit(nopython=True)
def dnu_dt_fitted(
    nu, I_syn, sigma, g_m=g_m, tau_m=tau_m, tau_rp=tau_rp
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

@jit(nopython=True)
def reweight_individual_xi(xi, W, mu):
    # pos_mask = xi > 0.
    for i in np.arange(xi.shape[0]):
        for j in np.arange(xi.shape[0]):
            if xi[i, j] > 0.:
                xi[i, j] *= (w_max_default - W[i, j])**mu
            elif xi[i, j] < 0.:
                xi[i, j] *= W[i, j]**mu
    # xi[pos_mask] = xi[pos_mask] * (w_max_default-W[pos_mask])**mu
    # xi[~pos_mask] = xi[~pos_mask] * W[~pos_mask]**mu
    return xi


@jit(nopython=True)
def get_xis_reweighted(
    xi_10_0, xi_10_1, xi_01_0, xi_01_1, xi_11_0, xi_11_1,
    xi_20_0, xi_20_1, xi_21_0, xi_21_1, xi_02_0, xi_02_1,
    xi_12_0, xi_12_1, W, mu
):
    xi_10_0 = np.full_like(W, xi_10_0)
    xi_10_0 = reweight_individual_xi(xi_10_0, W, mu)
    xi_10_1 = np.full_like(W, xi_10_1)
    xi_10_1 = reweight_individual_xi(xi_10_1, W, mu)
    xi_01_0 = np.full_like(W, xi_01_0)
    xi_01_0 = reweight_individual_xi(xi_01_0, W, mu)
    xi_01_1 = np.full_like(W, xi_01_1)
    xi_01_1 = reweight_individual_xi(xi_01_1, W, mu)
    xi_11_0 = np.full_like(W, xi_11_0)
    xi_11_0 = reweight_individual_xi(xi_11_0, W, mu)
    xi_11_1 = np.full_like(W, xi_11_1)
    xi_11_1 = reweight_individual_xi(xi_11_1, W, mu)
    xi_20_0 = np.full_like(W, xi_20_0)
    xi_20_0 = reweight_individual_xi(xi_20_0, W, mu)
    xi_20_1 = np.full_like(W, xi_20_1)
    xi_20_1 = reweight_individual_xi(xi_20_1, W, mu)
    xi_21_0 = np.full_like(W, xi_21_0)
    xi_21_0 = reweight_individual_xi(xi_21_0, W, mu)
    xi_21_1 = np.full_like(W, xi_21_1)
    xi_21_1 = reweight_individual_xi(xi_21_1, W, mu)
    xi_02_0 = np.full_like(W, xi_02_0)
    xi_02_0 = reweight_individual_xi(xi_02_0, W, mu)
    xi_02_1 = np.full_like(W, xi_02_1)
    xi_02_1 = reweight_individual_xi(xi_02_1, W, mu)
    xi_12_0 = np.full_like(W, xi_12_0)
    xi_12_0 = reweight_individual_xi(xi_12_0, W, mu)
    xi_12_1 = np.full_like(W, xi_12_1)
    xi_12_1 = reweight_individual_xi(xi_12_1, W, mu)
    return xi_10_0, xi_10_1, xi_01_0, xi_01_1, xi_11_0, xi_11_1,\
           xi_20_0, xi_20_1, xi_21_0, xi_21_1, xi_02_0, xi_02_1,\
           xi_12_0, xi_12_1
    

#region plasticity variables
@jit(nopython=True)
def H(nu, W, theta, plasticity_params):
    """Correct name..."""
    return F_full(nu, W, theta, plasticity_params)

# TODO: change name of F_full to match name in thesis
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

    # implementation 2: reweight xis before
    xi_10_0, xi_10_1, xi_01_0, xi_01_1, xi_11_0, xi_11_1,\
    xi_20_0, xi_20_1, xi_21_0, xi_21_1, xi_02_0, xi_02_1,\
    xi_12_0, xi_12_1 = get_xis_reweighted(
        xi_10_0, xi_10_1, xi_01_0, xi_01_1, xi_11_0, xi_11_1,
        xi_20_0, xi_20_1, xi_21_0, xi_21_1, xi_02_0, xi_02_1,
        xi_12_0, xi_12_1, W, mu)
        
    result += (xi_10_0 + xi_10_1 * theta_cast_p) * (nu @ ones_vec.T)
    result += (xi_01_0 + xi_01_1 * theta_cast_p) * (ones_vec @ nu.T)
    result += (xi_11_0 + xi_11_1 * theta_cast_p) * (nu @ nu.T)
    result += (xi_20_0 + xi_20_1 * theta_cast_p) * ((nu**2) @ ones_vec.T)
    result += (xi_21_0 + xi_21_1 * theta_cast_p) * ((nu**2) @ nu.T)
    result += (xi_02_0 + xi_02_1 * theta_cast_p) * (ones_vec @ (nu.T**2))
    result += (xi_12_0 + xi_12_1 * theta_cast_p) * (nu @ (nu.T**2))
    # result *= ((w_max_default - W)*W)**mu
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

# changed so that theta must be input
@jit(nopython=True)
def dtheta_dt(
    theta, nu,
    plasticity_params,
    # **kwargs
):
    """
    Threshold for the BCM-rule.
    
    tau_theta * dtheta/dt = -theta + nu**2
    
    Units of theta are technically different to those of nu. 
    There should be a constant to fix this.
    """
    # if theta is None:
    #     return None
    p_const, p_theta, mu, tau_theta, xi_00, \
    xi_10_0, xi_10_1, xi_01_0, xi_01_1, xi_11_0, xi_11_1, \
    xi_20_0, xi_20_1, xi_21_0, xi_21_1, xi_02_0, xi_02_1, \
    xi_12_0, xi_12_1, tau_e, beta = plasticity_params
    dtheta_dt_now = (-theta + nu)/tau_theta
    return dtheta_dt_now

# TODO: refactor so that F_val is given
@jit(nopython=True)
def de_dt(
    W, eligibility_trace, nu,
    plasticity_params,
    theta=None,
    F_val=None,
    # **kwargs
):
    p_const, p_theta, mu, tau_theta, xi_00, \
    xi_10_0, xi_10_1, xi_01_0, xi_01_1, xi_11_0, xi_11_1, \
    xi_20_0, xi_20_1, xi_21_0, xi_21_1, xi_02_0, xi_02_1, \
    xi_12_0, xi_12_1, tau_e, beta = plasticity_params
    # if F_val is None:
    #     F_val = F_full(nu, W, theta, plasticity_params)
    tau_de_dt = -eligibility_trace + F_val
    return tau_de_dt / tau_e


# TODO: refactor so that F_val is given
@jit(nopython=True)
def dW_dt(
    W, eligibility_trace, nu, reward,
    plasticity_params,
    theta=None,
    F_val=None,
    # **kwargs
):
    p_const, p_theta, mu, tau_theta, xi_00, \
    xi_10_0, xi_10_1, xi_01_0, xi_01_1, xi_11_0, xi_11_1, \
    xi_20_0, xi_20_1, xi_21_0, xi_21_1, xi_02_0, xi_02_1, \
    xi_12_0, xi_12_1, tau_e, beta = plasticity_params
    # if F_val is None:
    #     if beta > 0.:
    #         F_val = F_full(nu, W, theta, plasticity_params)
    #     else:
    #         F_val = 0.0
    tau_dw_dt = (1-beta)*eligibility_trace*reward + beta*F_val
    # plasticity_mask
    # for j, b in enumerate(plasticity_mask_source):
    #     if not b:
    #         tau_dw_dt[:, j] = 0.0
    # for i, b in enumerate(plasticity_mask_target):
    #     if not b:
    #         tau_dw_dt[i, :] = 0.0

    return tau_dw_dt  # / tau_w  # tau_w is redundant

#endregion

@jit(nopython=True)
def dR_dt_default(reward, tau_reward=tau_reward_default):
    dR_dt_now = -reward/tau_reward
    return dR_dt_now


#endregion

#region Simulation updates
@jit(nopython=True)
def update_dynamics_state_fitted(
    sigma, V_avg, s_AMPA_ext,
    nu, s_NMDA, s_AMPA, s_GABA, ic_noise,
    reward, W, eta
):
    """
    Updates dynamic variables state using default
    reward derivative and fitted function for phi.
    """
    # Compute inputs
    # TODO: factor out this product?
    ip_AMPA = (V_drive - V_E) * C_k * s_AMPA
    ic_AMPA = g_AMPA * (W @ ip_AMPA)

    g_NMDA_eff_V = g_NMDA_eff(V_avg)
    V_E_eff_V = V_E_eff(V_avg)
    ip_NMDA = (V_drive - V_E_eff_V) * C_k * s_NMDA
    ic_NMDA = g_NMDA_eff_V * (W @ ip_NMDA)

    ip_GABA = (V_drive - V_I) * C_k * s_GABA
    ic_GABA = g_GABA * (W @  ip_GABA)

    ip_AMPA_ext = (V_drive - V_E) * C_ext * s_AMPA_ext
    ic_AMPA_ext = g_AMPA_ext * ip_AMPA_ext

    I_syn = ic_AMPA + ic_AMPA_ext + ic_NMDA + ic_GABA + ic_noise

    # Compute derivatives
    dnu_dt_now = dnu_dt_fitted(
        nu=nu, I_syn=I_syn, sigma=sigma,
        # g_m=g_m,
        # tau_m=tau_m,
        # tau_rp=tau_rp
    )
    dic_noise_dt_now = dic_noise_dt_inputnoise(
        ic_noise, eta=eta
    )
    ds_NMDA_dt_now = ds_NMDA_dt(s_NMDA, nu)
    ds_AMPA_dt_now = ds_AMPA_dt(s_AMPA, nu)
    ds_GABA_dt_now = ds_GABA_dt(s_GABA, nu)
    dreward_dt_now = dR_dt_default(reward)

    # update dynamic states
    nu += dnu_dt_now * defaultdt
    ic_noise += dic_noise_dt_now * defaultdt
    s_NMDA += ds_NMDA_dt_now * defaultdt
    s_AMPA += ds_AMPA_dt_now * defaultdt
    s_GABA += ds_GABA_dt_now * defaultdt
    reward += dreward_dt_now * defaultdt


    V_SS = V_L - I_syn / g_m
    V_avg = V_SS - (V_thr-V_reset)*nu*tau_m - (V_SS-V_reset)*nu*tau_rp

    return I_syn, nu, s_NMDA, s_AMPA, \
        s_GABA, ic_noise, reward, V_avg

@jit(nopython=True)
def update_weight_state(
    nu,theta,e,W,reward,
    plasticity_params
):
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
    theta += dtheta_dt_now * defaultdt
    e += de_dt_now * defaultdt
    W_unclipped = W + dW_dt_now * defaultdt
    # W = np.clip(W, 0.0, w_max_default)
    return theta, e, W_unclipped

def compute_update_step(
    sigma, V_avg, nu, s_AMPA_ext, s_AMPA, s_NMDA, s_GABA,
    ic_noise, reward, W, theta, e,
    randomstate=random_state_default,
    plasticity=True,
    plasticity_params=nolearn_parameters,
    *args, **kwargs
):
    eta = randomstate.randn(*ic_noise.shape)
    I_syn_new, nu_new, s_NMDA_new, s_AMPA_new, \
    s_GABA_new, ic_noise_new, reward_new, V_avg_new = update_dynamics_state_fitted(
        sigma=sigma,
        V_avg=V_avg,
        s_AMPA_ext=s_AMPA_ext,
        nu=nu,
        s_NMDA=s_NMDA,
        s_AMPA=s_AMPA,
        s_GABA=s_GABA,
        ic_noise=ic_noise,
        reward=reward,
        W=W,
        eta=eta
    )
    if plasticity:
        theta_new, e_new, W_new_unclipped = update_weight_state(
            nu=nu,
            theta=theta,
            e=e,
            W=W,
            reward=reward,
            plasticity_params=plasticity_params
        )
        W_new = np.clip(W_new_unclipped, 0., w_max_default)
    else:
        theta_new, e_new, W_new = theta, e, W
    
    return I_syn_new, nu_new, s_NMDA_new, s_AMPA_new, \
        s_GABA_new, ic_noise_new, reward_new, V_avg_new, \
        theta_new, e_new, W_new
#endregion



# @jit(nopython=True)
# def dW_dt_BCM(W, nu, theta):
#     dW_dt = (np.outer(nu * (nu-theta), nu) / theta.reshape(-1, 1)) / tau_W
#     for j, b in enumerate(plasticity_mask_source):
#         if not b:
#             dW_dt[:, j] = 0.0
#     for i, b in enumerate(plasticity_mask_target):
#         if not b:
#             dW_dt[i, :] = 0.0
#     return dW_dt

#region Initialisation of weights
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
#endregion
