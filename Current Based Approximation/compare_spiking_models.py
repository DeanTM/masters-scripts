from parameters_spiking import *
import brian2 as b2
from brian2 import np, plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle

from datetime import datetime
from os import path, mkdir
import json

script_running_datetime = datetime.now()
folder_prefix = path.join(path.join('experiments', str(script_running_datetime)))
imagedir = path.join(folder_prefix, 'images_and_animations')
paramsdir = path.join(folder_prefix, 'parameters')
paramsfile = path.join(paramsdir, 'experiment_parameters.json')
namespacefile = path.join(paramsdir, 'experiment_namespace.json')


def plot_average_membrane_potential(
    axes, population='E_non', group='non-selective',
    tmin=None, tmax=None, legend=True, title=True,
    namespace=namespace
):
    # using formula:
    V_L = namespace['V_L']
    V_thr = namespace['V_thr']
    V_reset = namespace['V_reset']
    tau_m_E = namespace['tau_m_E']
    g_m_E = namespace['g_m_E']
    V_SS = V_L - net['st_'+population].I_syn.mean(axis=0) / (g_m_E)
    nu = net['r_'+population].smooth_rate('gaussian', 5*b2.ms)
    V_avg = V_SS - (V_thr - V_reset) * nu * tau_m_E
    # using data:
    v_mean = b2.mean(net['st_'+population].v, axis=0)
    v_std = b2.std(net['st_'+population].v, axis=0)
    
    if tmin is None:
        tmin = net['r_'+population].t[0]
    if tmax is None:
        tmax = net['r_'+population].t[-1]
    mask = (net['r_I'].t >= tmin) & (net['r_I'].t <= tmax)
    
    true_line = axes.plot(
        net['r_'+population].t[mask] / b2.ms,
        v_mean[mask] / b2.mV,
        label='true avg.',
        ls='--'
    )
    colour = true_line[0].get_color()
    axes.fill_between(
        net['r_'+population].t[mask] / b2.ms,
        (v_mean-v_std)[mask]/b2.mV,
        (v_mean+v_std)[mask]/b2.mV,
        alpha=0.6,
        label=r'$\pm 1$ std',
        color=colour
    )
    axes.fill_between(
        net['r_'+population].t[mask] / b2.ms,
        (v_mean-2*v_std)[mask]/b2.mV,
        (v_mean-v_std)[mask]/b2.mV,
        alpha=0.3,
        label=r'$\pm 2$ std',
        color=colour
    )
    axes.fill_between(
        net['r_'+population].t[mask] / b2.ms,
        (v_mean+2*v_std)[mask]/b2.mV,
        (v_mean+v_std)[mask]/b2.mV,
        alpha=0.3,
        color=colour
    )

    axes.plot(
        net['r_'+population].t[mask] / b2.ms,
        V_avg[mask] / b2.mV,
        label='computed',
        color='darkorange',
        lw=1
    )
    if legend:
        axes.legend()
    axes.grid(alpha=0.5)
    if title:
        axes.set_title(
            r'Comparison of $\langle V \rangle$ formula to true values'
            + f'\nGroup: {group}'
        )
    return None


def run_model(
    N=N, p=p, f=f, N_E=N_E, N_I=N_I, w_plus=w_plus, w_minus=w_minus,
    N_sub=N_sub, N_non=N_non, C_ext=C_ext, C_E=C_E, C_I=C_I,
    namespace=namespace, net=None,
    use_conductance=True,
    **namespace_kwargs  # use for repeated simulations?
):
    """
    Code taken primarily from
    https://brian2.readthedocs.io/en/stable/examples/frompapers.Brunel_Wang_2001.html
    """
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
        C_selection = int(f * C_ext)
        rate_selection = 25. * b2.Hz
        if 'stimulus1' not in namespace:
            stimtimestep = 25 * b2.ms
            stimtime = 1
            stimuli1 = b2.TimedArray(np.r_[
                np.zeros(8), np.ones(stimtime), np.zeros(100)],
                dt=stimtimestep
                )
            namespace['stimuli1'] = stimuli1
        input1 = b2.PoissonInput(
            P_E[N_non:N_non + N_sub],
            target_var='s_AMPA_ext',
            N=C_selection,
            rate=rate_selection,
            weight='stimuli1(t)'
        )

        N_activity_plot = 15  # number of neurons to rasterplot
        # spike monitors
        sp_E_sels = [
        b2.SpikeMonitor(P_E[pi:pi + N_activity_plot], name=f'sp_E_{int((pi-N_non)/N_sub) + 1}')
            for pi in range(N_non, N_non + p * N_sub, N_sub)
        ]
        sp_E = b2.SpikeMonitor(P_E[:N_activity_plot], name=f'sp_E_non')
        sp_I = b2.SpikeMonitor(P_I[:N_activity_plot], name=f'sp_I')
        # rate monitors
        r_E_sels = [
            b2.PopulationRateMonitor(P_E[pi:pi + N_sub], name=f'r_E_{int((pi-N_non)/N_sub) + 1}')
            for pi in range(N_non, N_non + p * N_sub, N_sub)]
        r_E = b2.PopulationRateMonitor(P_E[:N_non], name=f'r_E_non')
        r_I = b2.PopulationRateMonitor(P_I, name=f'r_I')
        # state monitors
        st_E_sels = [
            b2.StateMonitor(P_E[pi:pi + N_activity_plot], variables=True, record=True,
                        name=f'st_E_{int((pi-N_non)/N_sub) + 1}')
            for pi in range(N_non, N_non + p * N_sub, N_sub)]
        st_E = b2.StateMonitor(P_E[:N_activity_plot], variables=True, record=True, name=f'st_E_non')
        st_I = b2.StateMonitor(P_I[:N_activity_plot], variables=True, record=True, name=f'st_I')

        net = b2.Network(b2.collect())
        # add lists of monitors independently because
        # `b2.collect` doesn't search nested objects
        net.add(sp_E_sels)
        net.add(r_E_sels)
        net.add(st_E_sels)

        net.store('initialised')

    # runtime=runtime*b2.second
    net.restore('initialised')
    net.run(
        duration=runtime*b2.second,
        report='stdout',
        namespace=namespace
    )

    return net, namespace


if __name__ == '__main__':
    # Create folders and files
    namespace_nounits = dict(zip(
        namespace.keys(),
        map(lambda v: b2.array(v).item(), namespace.values())
        ))
    if not path.exists(folder_prefix):
        mkdir(folder_prefix)
    if not path.exists(imagedir):
        mkdir(imagedir)
    if not path.exists(paramsdir):
        mkdir(paramsdir)
    with open(paramsfile, 'w') as fp:
        json.dump(parameters_shared_dict, fp)
    with open(namespacefile, 'w') as fp:
        json.dump(namespace_nounits, fp)
    


    # Plot results
    V_drive_vals = [-55.*b2.mV, namespace['V_drive'], None]
    use_conductance_bools = [False, False, True]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    net = None
    for k in range(3):
        use_conductance = use_conductance_bools[k]
        V_drive = V_drive_vals[k]
        namespace_copy = dict(namespace)
        namespace_copy['V_drive'] = V_drive

        if k == 2:
            # reset for conductance model
            net = None

        net, namespace_copy = run_model(
            use_conductance=use_conductance,
            namespace=namespace_copy,
            net=None
        )
        st_E = net['st_E_non']
        v = st_E.v.reshape(-1) / b2.mV
        v_restricted = v[(v >= namespace['V_I']/ b2.mV) & (v <= namespace['V_E']/ b2.mV)]

        hist = axes[k,0].hist(v_restricted, bins=100, density=True, alpha=0.5)
        mean_restricted = np.mean(v_restricted)
        var_trestriced = np.var(v_restricted)
        axes[k,0].vlines(mean_restricted, 0.0, hist[0].max(), ls='--', label=f'mean')

        y_label = 'Conductance-Based' if use_conductance else 'Current-Based: $V_{drive}=$'+f"{V_drive*1000}mV"
        axes[k,0].set_ylabel(y_label)
        axes[k,0].grid(alpha=0.1)
        # axes[k,0].set_xlabel('Membrane Potential (mV)')
        if k == 0:
            axes[k,0].legend()
            axes[k,0].set_title('Membrane Potential Distribution (mV)')
        
        nu_E = net['r_E_non']
        smoothing_window = 5 * b2.ms
        axes[k,1].plot(
            nu_E.t / b2.ms,
            nu_E.smooth_rate('gaussian', smoothing_window),
            color='r', label='non-sel.')

        nu_E_sel = net['r_E_1']
        axes[k,1].plot(
            nu_E_sel.t / b2.ms,
            nu_E_sel.smooth_rate('gaussian', smoothing_window),
            color='g', label='sel.')

        nu_I = net['r_I']
        axes[k,1].plot(
            nu_I.t / b2.ms,
            nu_I.smooth_rate('gaussian', smoothing_window),
            color='b', label='inhib.')
        axes[k,1].legend()
        axes[k,1].set_xlabel('Time (ms)')
        # axes[k,1].set_ylabel(f'Firing Rate (Hz)\nsmoothed over {smoothing_window/b2.ms}ms')
        if k == 0:
            axes[k,1].set_title(f'Firing Rate of Selected Population (Hz)\nsmoothed over {smoothing_window/b2.ms}ms')
        axes[k,1].grid(alpha=0.1)
    plt.savefig(path.join(imagedir,'conductance_current_comparison.png'))
    plt.show()


    # Average membrane potential plot
    # use_conductance = True
    # net, namespace = run_model(
    #     use_conductance=use_conductance,
    #     namespace=namespace
    #     )
    fig, axes = plt.subplots(1,2,figsize=(14, 5), sharey=True)

    axes[0].set_xlabel('time (ms)')
    axes[1].set_xlabel('time (ms)')
    axes[0].set_ylabel('membrane potential (mV)')

    plot_average_membrane_potential(
        axes[0],
        tmax=400*b2.ms,
        namespace=namespace_copy
    )
    plot_average_membrane_potential(
        axes[1], population='E_1', group='selective-1',
        tmax=400*b2.ms,
        legend=False,
        namespace=namespace_copy
    )

    inset_ax = inset_axes(
        axes[1],
        width="40%", # width = 30% of parent_bbox
        height=1.2, # height : 1 inch
        loc=4,
        borderpad=2.0,
        axes_kwargs={'alpha':0.5}
    )
    plot_average_membrane_potential(
        inset_ax, population='E_1', group='selective-1',
        legend=False, title=False,
        tmin=190*b2.ms, tmax=240*b2.ms,
        namespace=namespace_copy
    )
    inset_ax.set_xticklabels([])
    inset_ax.set_yticklabels([])

    left, right = inset_ax.get_xlim()
    bottom, top = inset_ax.get_ylim()

    ax_ticksy = axes[1].get_yticks()
    ax_ticksy = ax_ticksy[ax_ticksy <= top]
    ax_ticksy = ax_ticksy[ax_ticksy >= bottom]
    inset_ax.set_yticks(ax_ticksy)

    ax_ticksx = axes[1].get_xticks()
    ax_ticksx = ax_ticksx[ax_ticksx <= right]
    ax_ticksx = ax_ticksx[ax_ticksx >= left]
    inset_ax.set_xticks(ax_ticksx)

    rectangle_patch = Rectangle(
        (left, bottom), right-left, top-bottom,
        fill=False, clip_on=False, alpha=0.9, ls='--'
        )

    axes[1].add_patch(rectangle_patch)

    fig.subplots_adjust(wspace=0.05)
    plt.savefig(path.join(
        imagedir, f'average-membrane-potential-comparison_conductance_{use_conductance}.png'
        ))
    plt.show()
