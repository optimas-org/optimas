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


Installation on a local computer
--------------------------------

The recommended approach is to install Optimas in a ``conda`` environment.

Install basic dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code::

    conda install numpy pandas

Install PyTorch
~~~~~~~~~~~~~~~

If your computer does not feature a CUDA-capable GPU, install PyTorch for CPU:

.. code::

    conda install pytorch cpuonly -c pytorch


If you have a CUDA-capable GPU and want to take make it available to Optimas,
install PyTorch with:

.. code::

    conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia

Install ``mpi4py``
~~~~~~~~~~~~~~~~~~
If your system has already an MPI implementation installed, install ``mpi4py``
using ``pip``:

.. code::

    pip install mpi4py

This will make sure that Optimas uses the existing MPI. The recommended
MPI implementation is MPICH.

If you don't have an existing MPI installation, the recommended approach is to
install ``mpi4py`` from ``conda``, including the MPI implementation corresponding
to your operating system.

On Linux and macOS:

.. code::

    conda install -c conda-forge mpi4py mpich

On Windows:

.. code::

    conda install -c conda-forge mpi4py msmpi

Install Optimas
~~~~~~~~~~~~~~~
Install the latest version directly from GitHub:

.. code::

    pip install git+https://github.com/optimas-org/optimas.git



Installation on HPC clusters
----------------------------

.. toctree::
   :maxdepth: 1

   installation_maxwell
   installation_juwels