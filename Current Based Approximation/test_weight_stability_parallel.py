from simulation import *
from time import time
from datetime import datetime

from dask import delayed, compute
import dask.multiprocessing
dask.config.set(scheduler='processes')

from os import path, mkdir, listdir
import json

import argparse

parser = argparse.ArgumentParser(
    description = 'testing feasible weight range'
)

parser.add_argument(
    '-w', '--w_max', type=float, default=3.5
    )
parser.add_argument(
    '-g', '--gridsize', type=int, default=3
    )
parser.add_argument(
    '-r', '--total_time', type=float, default=runtime
    )
parser.add_argument(
    '-l', '--lambda_', type=float, default=0.8
    )
parser.add_argument(
    '-p', '--plasticity',
    action='store_true'
)
parser.add_argument(
    '--nosave',
    action='store_true'
)
parser.add_argument(
    '--norun',
    action='store_true'
)
parser.add_argument(
    '-f', '--usefitted',
    action='store_true'
)
parser.add_argument(
    '--imagedir', type=str,
    default='images_and_animations'
)

args = parser.parse_args()

gridsize = args.gridsize
w_max = args.w_max
lambda_ = args.lambda_
plasticity = args.plasticity
use_phi_fitted = args.usefitted

total_time = args.total_time
stim_start = total_time
stim_end = total_time
eval_time = total_time

def run_test(
    w_plus=w_plus,
    w_minus=w_minus,
    **kwargs
):
    # generate weight matrix
    W = get_weights(w_plus=w_plus, w_minus=w_minus)
    # run simulation
    sim_start_time = time()
    if use_phi_fitted:
        results = run_trial_fitted(
            lambda_=lambda_,
            total_time=total_time,
            plasticity=plasticity,
            W=W,
            stim_start=stim_start,
            stim_end=stim_end,
            eval_time=eval_time,
        )
    else:
        results = run_trial(
            lambda_=lambda_,
            total_time=total_time,
            plasticity=plasticity,
            W=W,
            stim_start=stim_start,
            stim_end=stim_end,
            eval_time=eval_time,
            use_phi_fitted=use_phi_fitted
        )
    sim_end_time = time()
    with np.printoptions(precision=3, suppress=True):
        print(f'\nTotal simulation time taken: {sim_end_time - sim_start_time:.2f}s')
        print(f'Final firing rates:\n\t{results["nu"][:, -1]}')
        print(f'Param values:\n\tw_plus:{w_plus:.2f}, w_minus:{w_minus:.2f}')
    return results


run_test_delayed = delayed(run_test)


if __name__ == '__main__':
    if args.norun:
        print("Running no tests.")
        if not args.nosave:
            print("Loading most recent test results for plotting.")
            saved_firing_rates = sorted([
                x for x in listdir('stability_tests_data')
                if x.endswith('.npy')
                and x.startswith('stability-test-firing-rates')
                ])
            saved_grids = sorted([
                x for x in listdir('stability_tests_data')
                if x.endswith('.npy')
                and x.startswith('stability-test-grid')
                ])
            max_firing_rates = np.load(
                os.path.join(
                    'stability_tests_data',
                    saved_firing_rates[-1]))
            grid = np.load(
                os.path.join(
                    'stability_tests_data',
                    saved_grids[-1]))
            # hacky way to get the previous datetime
            save_datetime = saved_firing_rates[-1].split('rates-')[-1][:-4]
            gridsize = grid.shape[0]

    else:
        start = time()
        tests = []
        grid = np.linspace(0, w_max, gridsize)
        grid_list = list(grid)
        w_minus_vals, w_plus_vals = [], []
        for w_plus in grid:
            for w_minus in grid:
                w_plus_vals.append(w_plus)
                w_minus_vals.append(w_minus)
                tests.append(run_test_delayed(
                        w_plus=w_plus, w_minus=w_minus,
                        **vars(args)
                    ))
        print(f"Running a total of {len(tests)} tests")
        print(
            "General test params:"
            f"\n\ttotal_time:{total_time:.2f}s"
            f"\n\tlambda_:{lambda_:.2f}"
            f"\n\tw_max:{w_max:.2f}"
            f"\n\tfitted:{use_phi_fitted}"
            )
        tests_results = compute(*tests, traverse=False)
        end = time()
        print("\nTests complete!")
        print(f"Total time taken: {end-start:.2f}s")

        if not args.nosave:            
            save_datetime = str(datetime.now())

            max_firing_rates = np.empty(
            (grid.shape[0], grid.shape[0])
            )

            for w_plus, w_minus, results in zip(
                w_plus_vals, w_minus_vals, tests_results
            ):
                row = grid_list.index(w_plus)
                column = grid_list.index(w_minus)
                max_rate = results["nu"][:, -1].max()
                max_firing_rates[row, column] = max_rate
            
            np.save(
                path.join(
                    'stability_tests_data',
                    f'stability-test-firing-rates-{save_datetime}.npy'),
                max_firing_rates
                )
            np.save(
                path.join(
                    'stability_tests_data',
                    f'stability-test-grid-{save_datetime}.npy'),
                grid
                )        

    if not args.nosave:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter

        folder_prefix = path.join(path.join('experiments', save_datetime))
        imagedir = path.join(folder_prefix, args.imagedir)
        paramsdir = path.join(folder_prefix, 'parameters')
        paramsfile = path.join(paramsdir, 'experiment_parameters.json')
        if not path.exists(folder_prefix):
            mkdir(folder_prefix)
        if not path.exists(imagedir):
            mkdir(imagedir)
        # only save parameters if it is not a re-plot:
        if not args.norun:
            if not path.exists(paramsdir):
                mkdir(paramsdir)
            with open(paramsfile, 'w') as fp:
                json.dump(parameters_dict, fp)

        # save_datetime = str(datetime.now())

        fig, ax = plt.subplots(figsize=(7,6))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        img = ax.imshow(
            max_firing_rates[::-1,:],  # correct rotation of image
            cmap=plt.cm.jet,
            extent=[0., grid[-1], 0., grid[-1]],
            # alpha=0.8
            # vmax=min(200., np.max(max_firing_rates))
            )
        fig.colorbar(img)
        # for i in range(gridsize):
        #     for j in range(gridsize):
        #         plt.text(
        #             j, i,
        #             f"{max_firing_rates[i,j]:.2f}",
        #             ha="center", va="center", color="w"
        #             )

        n_ticks = 8 if gridsize >= 8 else gridsize
        labels = np.linspace(0, grid[-1], n_ticks)
        ax.set_xticks(
            # np.arange(0, gridsize, tick_step)
            labels
        )
        w_plus_plotvals = np.linspace(0, grid[-1], 100)
        w_minus_plotvals = 1.0 - f*(w_plus_plotvals - 1.0) / (1.0 - f) 
        ax.plot(w_minus_plotvals, w_plus_plotvals, ls='-.', color='white', label='starting curve')
        ax.plot([0., 2.1], [2.1, 2.1], ls=':', color='black', label='initial bounds')
        ax.plot([2.1, 2.1], [0., 2.1], ls=':', color='black')
        ax.legend(loc='upper right', facecolor='gray')
        ax.set_xticklabels(
            labels=labels,
            rotation=90
            )
        ax.set_yticks(
            # np.arange(0, gridsize, tick_step)
            labels
        )
        ax.set_yticklabels(
            labels=labels
            )
        ax.set_ylabel('$w_+$')
        ax.set_xlabel('$w_-$')
        ax.set_title(
            'Final Firing Rates\n' +\
            f"total time:{total_time:.2f}s " +\
            f"lambda_:{lambda_:.2f} " +\
            f"f:{f:.2f} " +\
            f"p:{p} " +\
            f"fitted:{use_phi_fitted} "
            )
        plt.tight_layout()
        plt.savefig(path.join(
            imagedir, f'firing_rate_stabilities.png'
            ))
