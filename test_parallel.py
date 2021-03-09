# TODO: this test file is not up to date,
# but no longer needed as parallelisation works
from functions import *
import matplotlib.pyplot as plt
from time import time
from datetime import datetime

from dask import delayed, compute
import dask.multiprocessing
dask.config.set(scheduler='processes')

import argparse

parser = argparse.ArgumentParser(
    description='testing parameters')
parser.add_argument(
    '-n', '--max_threads',
    type=int, default=60
)
parser.add_argument(
    '-s', '--savefig',
    action='store_true'
)
parser.add_argument(
    '-c', '--coherence',
    type=float, default=0.5
)
parser.add_argument(
    '-l', '--lambda_',
    type=float, default=0.8
)
parser.add_argument(
    '-p', '--plasticity',
    action='store_true'
)
parser.add_argument(
    '-f', '--usefitted',
    action='store_true'
)


args = parser.parse_args()

# proper `if __name__...` idiom needed for parallel processing
if __name__ == '__main__':        
    title_suffix = ''
    if args.savefig:
        for key,value in vars(args).items():
            if key not in ['savefig', 'max_threads']:
                title_suffix += f'{key}:{value} '
        title_suffix += f'\np:{p} w+:{w_plus} fitted:{args.usefitted}'
        
    
    sim_delayed = delayed(simulate_original)
    max_threads = args.max_threads
    times = np.empty(max_threads, dtype=float)
    for i in range(1, max_threads+1):
        print(f"itr {i}/{max_threads}:", end='\t')
        start = time()
        computations = [
            sim_delayed(
                coherence=args.coherence,
                lambda_=args.lambda_,
                plasticity=args.plasticity,
                use_phi_fitted=args.usefitted
            ) for k in range(i)
        ]
        result = compute(computations)
        end = time()
        times[i-1] = end-start
        print(f"time taken: {times[i-1]:.2f} seconds")
    if args.savefig:
        plt.plot(np.arange(1, max_threads+1), times)
        plt.xlabel('number of processes')
        plt.ylabel('time taken (seconds)')
        plt.title(
            f'Parallelisation Speeds\n' + title_suffix)
        plt.savefig(
            f'parallelisation_speeds_{datetime.now()}.png'
        )
    