from parameters_shared import *
import brian2 as b2
from brian2 import asarray


# namespace has units for brian2 spiking simulation
namespace = dict(
    V_L = -70. * b2.mV,
    V_thr = -50. * b2.mV,
    V_reset = -55. * b2.mV,
    V_E = 0. * b2.mV,
    V_I = -70. * b2.mV,
    V_drive = -47.5 * b2.mV,
    C_m_E = 0.5 * b2.nF,
    C_m_I = 0.2 * b2.nF,
    g_m_E = 25. * b2.nS,
    g_m_I = 20. * b2.nS,
    tau_rp_E = 2. * b2.ms,
    tau_rp_I = 1. * b2.ms,
    g_AMPA_ext_E = 2.08 * b2.nS,
    g_AMPA_rec_E = 0.104 * b2.nS * 800. / N_E,
    g_AMPA_ext_I = 1.62 * b2.nS,
    g_AMPA_rec_I = 0.081 * b2.nS * 800. / N_E,
    tau_AMPA = 2. * b2.ms,
    g_NMDA_E = 0.327 * b2.nS * 800. / N_E,
    g_NMDA_I = 0.258 * b2.nS * 800. / N_E,
    tau_NMDA_rise = 2. * b2.ms,
    tau_NMDA_decay = 100. * b2.ms,
    alpha = 0.5 / b2.ms,
    Mg2 = 1.,
    g_GABA_E = 1.25 * b2.nS * 200. / N_I,
    g_GABA_I = 0.973 * b2.nS * 200. / N_I,
    tau_GABA = 10. * b2.ms,
    rate_ext = rate_ext * b2.Hz
)

eqs_conductance_E = '''
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

eqs_conductance_I = '''
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

eqs_current_E = '''
    dv / dt = (- g_m_E * (v - V_L) - I_syn) / C_m_E : volt (unless refractory)

    I_syn = I_AMPA_ext + I_AMPA_rec + I_NMDA_rec + I_GABA_rec : amp

    I_AMPA_ext = g_AMPA_ext_E * (V_drive - V_E) * s_AMPA_ext : amp
    I_AMPA_rec = g_AMPA_rec_E * (V_drive - V_E) * 1 * s_AMPA : amp
    ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA : 1
    ds_AMPA / dt = - s_AMPA / tau_AMPA : 1

    I_NMDA_rec = g_NMDA_E * (V_drive - V_E) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) * s_NMDA_tot : amp
    s_NMDA_tot : 1

    I_GABA_rec = g_GABA_E * (V_drive + 2*mV - V_I) * s_GABA : amp
    ds_GABA / dt = - s_GABA / tau_GABA : 1
        '''

eqs_current_I = '''
    dv / dt = (- g_m_I * (v - V_L) - I_syn) / C_m_I : volt (unless refractory)

    I_syn = I_AMPA_ext + I_AMPA_rec + I_NMDA_rec + I_GABA_rec : amp

    I_AMPA_ext = g_AMPA_ext_I * (V_drive - V_E) * s_AMPA_ext : amp
    I_AMPA_rec = g_AMPA_rec_I * (V_drive - V_E) * 1 * s_AMPA : amp
    ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA : 1
    ds_AMPA / dt = - s_AMPA / tau_AMPA : 1

    I_NMDA_rec = g_NMDA_I * (V_drive - V_E) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) * s_NMDA_tot : amp
    s_NMDA_tot : 1

    I_GABA_rec = g_GABA_I * (V_drive + 2*mV - V_I) * s_GABA : amp
    ds_GABA / dt = - s_GABA / tau_GABA : 1
        '''

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
# are these next ones needed?
# they'd be used for plasticity
eqs_post_glut = ''