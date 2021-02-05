from parameters_shared import *

defaultdt = 0.05 * 1e-3

V_avg_initial = -56e-3
V_L = -70. * 1e-3
V_thr = -50. * 1e-3
V_reset = -55. * 1e-3
V_E = 0. * 1e-3  # shouldn't be used
V_I = -70 * 1e-3  # shouldn't be used
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
rate_ext = 3.

# firing rate dynamics time constant
tau_r = 2e-3

# initial values
rate_interneuron = 5.
rate_pyramidal = 3.