.. _dependencies:

Dependencies
============

Optimas relies on the following packages:

* `jinja2 <https://pypi.org/project/jinja2/>`_ - Needed to generate simulation scripts from templates.
* `libEnsemble <https://pypi.org/project/libensemble/>`_ - The backbone of optimas, orchestrates the concurrent evaluation of simulations, the resource detection and allocation, and the communication between simulations and manager.
* `mpi4py <https://pypi.org/project/mpi4py/>`_ - Python bindings for MPI. Required for launching parallel simulations.
* `pandas <https://pypi.org/project/pandas/>`_ - Enable output as pandas DataFrames.
* `pydantic <https://pypi.org/project/pydantic/>`_ - Input validation and object serialization.
* (optional) `Ax <https://pypi.org/project/ax-platform/>`_ - Algorithms for Bayesian optimization.


The installed dependencies will determine which generators are available for use.
See table below for a summary.

.. list-table:: Available generators and their dependencies
   :widths: 35 25 25
   :header-rows: 1

   * - Generator
     - ``pip install optimas``
     - ``pip install 'optimas[all]'``
   * - :class:`~optimas.generators.LineSamplingGenerator`
     - :math:`\checkmark`
     - :math:`\checkmark`
   * - :class:`~optimas.generators.GridSamplingGenerator`
     - :math:`\checkmark`
     - :math:`\checkmark`
   * - :class:`~optimas.generators.RandomSamplingGenerator`
     - :math:`\checkmark`
     - :math:`\checkmark`
   * - :class:`~optimas.generators.AxSingleFidelityGenerator`
     - :math:`\times`
     - :math:`\checkmark`
   * - :class:`~optimas.generators.AxMultiFidelityGenerator`
     - :math:`\times`
     - :math:`\checkmark`
   * - :class:`~optimas.generators.AxMultitaskGenerator`
     - :math:`\times`
     - :math:`\checkmark`
   * - :class:`~optimas.generators.AxClientGenerator`
     - :math:`\times`
     - :math:`\checkmark`
