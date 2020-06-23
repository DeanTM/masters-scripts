# A file to create simulation instances, which can run the simulation
# with it's own parameters. The idea is to generate simulations for the 
# evolutionary algorithm to evaluate parameters.
import brian2 as b2
from brian2 import numpy as np

from simulation_equations import *
from simulation_config import *


def sample_weights(
    plus_minus_else, num_weights, std, lognormal_weights, w_min, w_max
):
    """Samples initial weights for the synapses."""
    pass
    # if plus_minus_else == 'plus':
    #     loc=w_plus
    # elif plus_minus_else == 'minus':
    #     loc=w_minus
    # elif plus_minus_else == 'else':
    #     loc=w_else
    # else:
    #     raise Exception(f'{plus_minus_else} not valid weight type')
    # if lognormal_weights:
    #     # mean is set to be equal to loc (lognormal takes mean of underlying normal as input)
    #     # while std is std of underlying normal
    #     weights = np.random.lognormal(mean=(np.log(loc) - 0.5*std**2), sigma=std, size=num_weights)
    # else:
    #     weights = np.random.normal(loc=loc, scale=std, size=num_weights)
    # weights[weights < w_min] = w_min
    # weights[weights > w_max] = w_max
    # return weights

class Simulation():
    def __init__(self, *args, **kwargs):
        """Initialises the network
        pass
        # self.params = DEFAULT_PARAMS
        # self.params.update(kwargs)

        # neuron_group_excitatory = b2.NeuronGroup(
        #     N=self.params['N_E'],
        #     model=neuron_eqns_excitatory,  # TODO: Handle membrane noise equations
        #     method='euler',  # TODO: allow for different methods
        #     threshold=threshold_eqn,
        #     reset=reset_eqn,
        #     refractory=self.params['tau_refrac_E']
        # )
        # neuron_group_excitatory.u = np.random.uniform(
        #     low=self.params['V_reset']/b2.mV,
        #     high=self.params['V_thresh']/b2.mV,
        #     size=self.params['N_E']
        # ) * b2.mV
        # N_sel = self.params['N_sel']
        # # selective_neuron_groups = []
        # neuron_group_1 = neuron_group_excitatory[:N_sel]
        # neuron_group_2 = neuron_group_excitatory[N_sel:2*N_sel]
        # neuron_group_other = neuron_group_excitatory[2*N_sel:]



if __name__ == '__main__':
    sim = Simulation()
    # print(*sim.params.items(), sep='\n')