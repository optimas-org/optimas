Dependencies
============

Optimas relies on the following packages:

* `mpi4py <https://pypi.org/project/mpi4py/>`_ - Python bindings for MPI. Required for launching parallel simulations.
* `libEnsemble <https://pypi.org/project/libensemble/>`_ - The backbone of optimas, orchestrates the concurrent evaluation of simulations, the resource detection and allocation, and the communication between simulations and manager.
* `jinja2 <https://pypi.org/project/jinja2/>`_ - Needed to generate simulation scripts from templates.
* `Ax <https://pypi.org/project/ax-platform/>`_ - Algorithms for Bayesian optimization.
