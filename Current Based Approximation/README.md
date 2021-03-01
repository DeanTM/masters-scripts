This constitutes the majority of the work on my masters project, entitled "Biologically Motivated Reinforcement Learning in Spiking Neural Networks."  

The code doesn't follow PEP8 guidelines and is unsafely written with star imports. The project structure is as follows (in order of imports):

* [`parameters_shared.py`](./parameters_shared.py) is the root of all the files, in which parameters shared for both the spiking and rate-based models are kept.
* [`parameters_spiking.py`](./parameters_spiking.py) imports from [`parameters_shared.py`](./parameters_shared.py) and ends the chain of import for the spiking model code.  
* [`parameters.py`](./parameters.py) contains the parameters and some helpful constructs, including prespecified learning rules and vectorisations of the network parameters. It inherits from [`parameters_shared.py`](./parameters_shared.py) for many of these constructions.  
* [`functions.py`](./functions.py) begins the functionality of the rate-based model. It inherits [`parameters.py`](./parameters.py) and specifies `numba.jit`ed functions for fast computation of the rate-based model.
* [`simulation.py`](./simulation.py) inherits from [`functions.py`](./functions.py) and describes the simulations which will be run as tasks. Effectively it brings the functions in [`functions.py`](./functions.py) together into cohesive units.  
* [`evolution.py`](./evolution.py) ends the chain, inheriting from [`simulation.py`](./simulation.py). Here, classes and functions useful for the evolutionary algorithm are defined.

Aside from these `.py` files mentioned, there are also several scripts which run various evolutionary tasks, test stability of the code, or plot the plots of the thesis. There may remain some `.ipynb` Jupyter notebooks which were used originally in creating the plots or fitting some functions. They were not updated as the previous files were and as such are fairly untidy.

Many of the scripts require the presence of a folder named `experiments` and/or a folder named `images_and_animations`. Neither of these folders are automatically created, but are used to store the plots and results of various functions.

Finally, the fitted version of the firing rate function, `phi_fitted` in [`functions.py`](./functions.py) depends on fitted parameters. Because these parameters are loaded outside the function and within the scope of downstream imports, unless those parameters are provided as `.npy` files, the code wil not run. These can be generated using the final method in [`Approximating-Phi.ipynb`](./`Approximating-Phi.ipynb`).
