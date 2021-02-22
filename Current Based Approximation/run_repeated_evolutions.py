from subprocess import Popen
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights', action='store_true')

args = parser.parse_args()
script = 'find_optimal_weights.py' if args.weights else 'run_evolution.py'

coherence = np.random.rand()
kwargs = ["--n_gen", "200",
    "--n_runs", "100",
    "--n_multiples", "10",
    "--coherence", str(coherence)
    ]
p = Popen([
    "python",
    script,
    *kwargs
])