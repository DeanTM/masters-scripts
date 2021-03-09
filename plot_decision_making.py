#FIXME: doesn't seem to work right. Maybe parameters are off
# Either reverbratory activity is too strong or non-existent
from parameters_spiking import *
import brian2 as b2
from brian2 import np, plt
from datetime import datetime
from os import path, mkdir

script_running_datetime = str(datetime.now()).replace(' ', '_')
folder_name = '_'.join([__file__[:-3], script_running_datetime])
folder_prefix = path.join(path.join('experiments', folder_name))
imagedir = path.join(folder_prefix, 'images_and_animations')
paramsdir = path.join(folder_prefix, 'parameters')
paramsfile = path.join(paramsdir, 'experiment_parameters.json')
namespacefile = path.join(paramsdir, 'experiment_namespace.json')


def run_model(
    N=N, p=p, f=f, N_E=N_E, N_I=N_I, w_plus=w_plus, w_minus=w_minus,
    N_sub=N_sub, N_non=N_non, C_ext=C_ext, C_E=C_E, C_I=C_I,
    namespace=namespace, net=None,
    use_conductance=True,
    coherence=0.2,
    stim_on=100*b2.ms, stim_off=900*b2.ms,
    runtime=2*b2.second,
    **namespace_kwargs  # use for repeated simulations?
):
    if net is None:
        b2.start_scope()
        s_AMPA_initial_ext = namespace['rate_ext'] * C_ext * namespace['tau_AMPA']

        # update namespace with computed variables
        namespace['tau_m_E'] = namespace['C_m_E'] / namespace['g_m_E']
        namespace['tau_m_I'] = namespace['C_m_I'] / namespace['g_m_I']


        P_E = b2.NeuronGroup(
            N_E,
            eqs_conductance_E if use_conductance else eqs_current_E,
            threshold='v > V_thr',
            reset='v = V_reset',
            refractory='tau_rp_E',
            method='euler',
            name='P_E'
        )
        P_E.v = namespace['V_L']
        P_E.s_AMPA_ext = s_AMPA_initial_ext  # estimated 4.8

        P_I = b2.NeuronGroup(
            N_E,
            eqs_conductance_I if use_conductance else eqs_current_I,
            threshold='v > V_thr',
            reset='v = V_reset',
            refractory='tau_rp_I',
            method='euler',
            name='P_I'
        )
        P_I.v = namespace['V_L']
        P_I.s_AMPA_ext = s_AMPA_initial_ext

        C_E_E = b2.Synapses(
            P_E, P_E,
            model=eqs_glut,  # equations for NMDA
            on_pre=eqs_pre_glut,
            on_post=eqs_post_glut,
            method='euler',
            name='C_E_E'
            )
        C_E_E.connect('i != j')
        C_E_E.w[:] = 1.0

        for pi in range(N_non, N_non+p*N_sub, N_sub):
            # internal other subpopulation to current nonselective
            # brian synapses are i->j
            C_E_E.w[C_E_E.indices[:, pi:pi+N_sub]] = w_minus
            # internal current subpopulation to current subpopulation
            C_E_E.w[C_E_E.indices[pi:pi + N_sub, pi:pi + N_sub]] = w_plus
        
        C_E_I = b2.Synapses(
            P_E, P_I,
            model=eqs_glut,
            on_pre=eqs_pre_glut,
            on_post=eqs_post_glut,
            method='euler',
            name='C_E_I'
        )
        C_E_I.connect()
        C_E_I.w[:] = 1.0

        C_I_I = b2.Synapses(
            P_I, P_I,
            on_pre=eqs_pre_gaba,
            method='euler',
            name='C_I_I'
        )
        C_I_I.connect('i != j')

        C_I_E = b2.Synapses(
            P_I, P_E,
            on_pre=eqs_pre_gaba,
            method='euler',
            name='C_I_E'
            )
        C_I_E.connect()

        C_P_E = b2.PoissonInput(
            P_E,
            target_var='s_AMPA_ext',
            N=C_ext,
            rate=namespace['rate_ext'],
            weight=1.
        )
        C_P_I = b2.PoissonInput(
            P_I,
            target_var='s_AMPA_ext',
            N=C_ext,
            rate=namespace['rate_ext'],
            weight=1.
        )


        # TODO: change the stimulus to match the task
        # C_selection = int(f * C_ext)
        # rate_selection = 25. * b2.Hz
        # if 'stimulus1' not in namespace:
        #     stimtimestep = 25 * b2.ms
        #     stimtime = 1
        #     stimuli1 = b2.TimedArray(np.r_[
        #         np.zeros(8), np.ones(stimtime), np.zeros(100)],
        #         dt=stimtimestep
        #         )
        #     namespace['stimuli1'] = stimuli1
        # input1 = b2.PoissonInput(
        #     P_E[N_non:N_non + N_sub],
        #     target_var='s_AMPA_ext',
        #     N=C_selection,
        #     rate=rate_selection,
        #     weight='stimuli1(t)'
        # )

        N_input = C_ext
        increment1 = 0.05*(1 + coherence) * namespace['rate_ext']
        increment2 = 0.05*(1 - coherence) * namespace['rate_ext']
        sigma_rate = 0.05 * namespace['rate_ext']

        input1 = b2.PoissonGroup(
            N_input, rates=0. * b2.Hz
        )
        input1_syn = b2.Synapses(
            input1, P_E[N_non:N_non + N_sub],
            model='',
            on_pre='s_AMPA_ext_post += 1'
        )
        input1_syn.connect()

        input2 = b2.PoissonGroup(
            N_input, rates=0. * b2.Hz
        )
        input2_syn = b2.Synapses(
            input2, P_E[N_non+N_sub:N_non + 2*N_sub],
            model='',
            on_pre='s_AMPA_ext_post += 1'
        )
        input2_syn.connect()

        @b2.network_operation(dt=50*b2.ms, when='start')
        def update_inputs(t):
            if t < stim_on or t >= stim_off:
                input1.rates = 0. * b2.Hz
                input2.rates = 0. * b2.Hz
            else:
                input1.rates = (np.random.randn()*sigma_rate + increment1)
                # input1.rates = increment1
                input2.rates = (np.random.randn()*sigma_rate + increment2)
                # input2.rates = increment2
        

        # ri1 = b2.PopulationRateMonitor(input1_syn, name='ri1')
        # ri2 = b2.PopulationRateMonitor(input2_syn, name='ri2')
        r0 = b2.PopulationRateMonitor(P_E[:N_non], name='r0')
        r1 = b2.PopulationRateMonitor(P_E[N_non:N_non + N_sub], name='r1')
        r2 = b2.PopulationRateMonitor(P_E[N_non+N_sub:N_non + 2*N_sub], name='r2')
        rI = b2.PopulationRateMonitor(P_I, name='rI')
        net = b2.Network(b2.collect())
        net.store('initialised')
    
    net.restore('initialised')
    net.run(
        duration=runtime,
        report='stdout',
        namespace=namespace
    )
    return net

if __name__ == '__main__':
    net = None
    N_traces = 5
    conv_width = 10*b2.ms
    leaveout_steps = int(conv_width/b2.defaultclock.dt)
    # leaveout_steps = 10
    b2.figure()
    for trace in range(N_traces):
        net = run_model(net=net)
        r1 = net['r1']
        r2 = net['r2']
        r0 = net['r0']
        rI = net['rI']
        # ri1 = net['ri1']
        # ri2 = net['ri2']

        b2.plot(
            r1.smooth_rate(width=conv_width)[:-leaveout_steps]/b2.Hz,
            r2.smooth_rate(width=conv_width)[:-leaveout_steps]/b2.Hz
            )
    ymin, ymax = b2.ylim()
    xmin, xmax = b2.xlim()
    b2.ylim([min(xmin, ymin), max(xmax, ymax)])
    b2.xlim([min(xmin, ymin), max(xmax, ymax)])

    b2.figure()
    b2.plot(r1.t[:-leaveout_steps]/b2.ms, 
        r1.smooth_rate(width=conv_width)[:-leaveout_steps]/b2.Hz, label='1')
    b2.plot(r2.t[:-leaveout_steps]/b2.ms, 
        r2.smooth_rate(width=conv_width)[:-leaveout_steps]/b2.Hz, label='2')
    b2.plot(r0.t[:-leaveout_steps]/b2.ms, 
        r0.smooth_rate(width=conv_width)[:-leaveout_steps]/b2.Hz, label='0')
    b2.plot(rI.t[:-leaveout_steps]/b2.ms, 
        rI.smooth_rate(width=conv_width)[:-leaveout_steps]/b2.Hz, label='I')
    # b2.plot(ri1.t/b2.ms, ri1.smooth_rate(width=10*b2.ms)/b2.Hz, label='i1')
    # b2.plot(ri2.t/b2.ms, ri2.smooth_rate(width=10*b2.ms)/b2.Hz, label='i2')
    b2.legend(title='pop.')
    b2.show()