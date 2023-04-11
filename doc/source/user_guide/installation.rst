Installing Optimas
==================
Optimas is supported on Python 3.8 and above. The package can be installed from
PyPI using ``pip`` or directly from GitHub.


Dependencies
------------
Optimas relies on the following packages:

* `NumPy <https://pypi.org/project/numpy/>`_ - Basic dependency for array operations.
* `Pandas <https://pypi.org/project/pandas/>`_ - Data analysis library required for post-processing and other dependencies.
* `mpi4py <https://pypi.org/project/mpi4py/>`_ - Python bindings for MPI. Required for launching parallel simulations.
* `libEnsemble <https://pypi.org/project/libensemble/>`_ - The backbone of Optimas, orchestrates the concurrent evaluation of simulations, the resource detection and allocation, and the communication between simulations and manager.
* `jinja2 <https://pypi.org/project/jinja2/>`_ - Needed to generate simulation scripts from templates.
* `Pytorch <https://pypi.org/project/pytorch/>`_ - Required by the Bayesian optimization generators.
* `Ax <https://pypi.org/project/ax-platform/>`_ - Algorithms for Bayesian optimization.

Instructions
------------
Step-by-step instructions for installing Optimas on a local computer and on
different HPC clusters:

.. toctree::
   :maxdepth: 1

   installation_local
   installation_maxwell
   installation_juwels