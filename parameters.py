from parameters_shared import *
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence

#region Simulation and Function Params
# too small, and the reward doesn't fully accumulate
# defaultdt = 0.05e-3  
# defaultdt = 0.1e-3
defaultdt = 0.25e-3


V_L = -70. * 1e-3
V_thr = -50. * 1e-3
V_reset = -55. * 1e-3
V_E = 0. * 1e-3
V_I = -70 * 1e-3
V_drive = -47.5 * 1e-3
V_avg_initial = -52.5 * 1e-3  # is this needed?
g_AMPA_ext_E = 2.08 * 1e-9 * 0.5
g_AMPA_rec_E = 0.104 * 1e-9 * 800. / N_E
g_AMPA_ext_I = 1.62 * 1e-9 * 0.5
g_AMPA_rec_I = 0.081 * 1e-9 * 800. / N_E
tau_AMPA = 2. * 1e-3
g_NMDA_E = 0.327 * 1e-9 * 800. / N_E
g_NMDA_I = 0.258 * 1e-9 * 800. / N_E
tau_NMDA_rise = 2. * 1e-3
tau_NMDA_decay = 100. * 1e-3
alpha = 0.5 / 1e-3
tau_NMDA = tau_NMDA_rise * alpha * tau_NMDA_decay
gamma_JahrStevens = 1. / 3.57
beta_JahrStevens = 0.062 / 1e-3
g_GABA_E = 1.25 * 1e-9 * 200. / N_I
g_GABA_I = 0.973 * 1e-9 * 200. / N_I
tau_GABA = 10. * 1e-3
tau_rp_E = 2. * 1e-3
tau_rp_I = 1. * 1e-3
C_m_E = 0.5 * 1e-9
C_m_I = 0.2 * 1e-9
g_m_E = 25. * 1e-9
g_m_I = 20. * 1e-9
tau_m_E = C_m_E / g_m_E
tau_m_I = C_m_I / g_m_I

# synaptic noise
sigma_noise=7e-12 # should this scale with number of populations?

# firing rate dynamics time constant
tau_rate = 2e-3

# initial values
rate_interneuron = 5.
rate_pyramidal = 3.

# param vectors
C_k = np.array([N_non] + [N_sub] * p + [N_I])
g_m = np.array([g_m_E] * (p+1) + [g_m_I])
C_m = np.array([C_m_E] * (p+1) + [C_m_I])
tau_m = np.array([tau_m_E] * (p+1) + [tau_m_I])
tau_rp = np.array([tau_rp_E] * (p+1) + [tau_rp_I])
nu_initial = np.array([rate_pyramidal] * (p+1) + [rate_interneuron])

g_AMPA = np.array([g_AMPA_rec_E]* (p+1) + [g_AMPA_rec_I])
g_AMPA_ext = np.array([g_AMPA_ext_E]* (p+1) + [g_AMPA_ext_I])
g_GABA = np.array([g_GABA_E]* (p+1) + [g_GABA_I])
g_NMDA = np.array([g_NMDA_E]* (p+1) + [g_NMDA_I])

# randomness
# default is given for function defaults
random_state_default = RandomState(MT19937(SeedSequence(1337)))

#endregion

#region Plasticity Params
w_max_default = 3.5
mu_default = 0.2

# reward defaults
tau_reward_default = 1. * 1e-3
# tau_e_default = 1. # not used
# beta_default = 0. # not used
# tau_W = 10.  # not used, redundant

# learning rules
# weight params vector will be of the form
# (p_const, p_theta, mu, tau_theta, xi_00,
#  xi_10_0, xi_10_1, xi_01_0, xi_01_1, xi_11_0, xi_11_1, 
#  xi_20_0, xi_20_1, xi_21_0, xi_21_1, xi_02_0, xi_02_1,
#  xi_12_0, xi_12_1, tau_e, beta)
eta_Oja = 0.5
Oja_parameters = (
    2.,1.,0.,tau_rate,eta_Oja,
    0.,0.,0.,0.,eta_Oja,0.,
    0.,0.,0.,0.,0.,0.,
    0.,0.,1.,1.
)
BCM_parameters = (
    0.,2.,0.,10.,0.,
    0.,0.,0.,0.,0.,
    -0.0337/64., # ??? any negative number, but what scale
    0.,0., 0.0337*0.0168, 0.,0.,0.,
    0.,0.,1.,1.
)

# beta to reward-only learning
nolearn_parameters = (
    0.,1.,0.5,100.,0.,
    0.,0.,0.,0.,0.,0.,
    0.,0.,0.,0.,0.,0.,
    0.,0.,100.,0.
)
#endregion

#region Evolution Params

# beta to reward-only learning
nolearn_genome = [
    0.,-10.,0.,100.,0.,0.,0.,0.,0.,0.,
    0.,0.,0.,0.,0.,0.,0.,0.,0.,100.,-10.
]

#endregion

# param dict for reference
parameters_dict = dict(
    N=N,
    N_E=N_E,
    N_I=N_I,
    f=f,
    p=p,
    N_sub=N_sub,
    N_non=N_non,
    w_plus=w_plus,
    w_minus=w_minus,
    C_ext=C_ext,
    C_E=C_E,
    C_I=C_I,
    rate_ext=rate_ext,
    runtime=runtime,
    defaultdt=defaultdt,
    V_L=V_L,
    V_thr=V_thr,
    V_E=V_E,
    V_I=V_I,
    V_drive=V_drive,
    V_avg_initial=V_avg_initial,
    g_AMPA_ext_E=g_AMPA_ext_E,
    g_AMPA_rec_E=g_AMPA_rec_E,
    g_AMPA_ext_I=g_AMPA_ext_I,
    g_AMPA_rec_I=g_AMPA_rec_I,
    tau_AMPA=tau_AMPA,
    g_NMDA_E=g_NMDA_E,
    g_NMDA_I=g_NMDA_I,
    tau_NMDA_rise=tau_NMDA_rise,
    tau_NMDA_decay=tau_NMDA_decay,
    alpha=alpha,
    tau_NMDA=tau_NMDA,
    gamma_JahrStevens=gamma_JahrStevens,
    beta_JahrStevens=beta_JahrStevens,
    g_GABA_E=g_GABA_E,
    g_GABA_I=g_GABA_I,
    tau_GABA=tau_GABA,
    tau_rp_E=tau_rp_E,
    tau_rp_I=tau_rp_I,
    C_m_E=C_m_E,
    C_m_I=C_m_I,
    g_m_E=g_m_E,
    g_m_I=g_m_I,
    tau_m_E=tau_m_E,
    tau_m_I=tau_m_I,
    sigma_noise=sigma_noise,
    tau_rate=tau_rate,
    rate_interneuron=rate_interneuron,
    rate_pyramidal=rate_pyramidal,
    w_max_default=w_max_default,
    mu_default=mu_default,
    tau_reward_default=tau_reward_default
)