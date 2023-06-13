Optimization with FBPIC
=======================


Description
~~~~~~~~~~~

This examples shows how to perform a Bayesian optimization of a laser-plasma
accelerator (LPA) using FBPIC simulations.

The LPA to be optimized is based on the LUX design [1]_ using ionization
injection.

The objective function to optimize (maximize) is defined as

.. math::

   f = \frac{\sqrt{Q} E_{MED}}{100 E_{MAD}}


where :math:`Q` is the beam charge, :math:`E_{MED}` is the median energy, and
:math:`E_{MAD}` is the median absolute deviation energy spread. This objective
is optimized by tuning 4 parameters:

- ``'laser_scale'``: parameter in the range :math:`[0.7, 1.05]` that scales
  the energy of the laser, which for ``laser_scale=1`` is
  :math:`2.56 \, \mathrm{J}`.
- ``'z_foc'``: the focal position of the laser in millimetres, with range
  :math:`[3, 7.5]`.
- ``'mult'``: parameter in the range :math:`[0.1, 1.5]` that scales the
  concentration of nitrogen in the injection region.
- ``'plasma_scale'``: parameter in the range :math:`[0.6, 0.8]` that scales
  the plasma density of all species.

The optimization is carried out using an
:class:`~optimas.generators.AxSingleFidelityGenerator` and a
:class:`~optimas.evaluators.TemplateEvaluator`. In this case, the function
``analyze_simulation`` that analyzes the output of each simulation is defined
in a separate file ``analysis_script.py`` and imported into the main
optimas script.

The example is set up to make use of a system of 4 GPUs, where each FBPIC
simulation uses a single GPU and 4 simulations are carried out in parallel.


Scripts
~~~~~~~

The files needed to run the optimization should be located in a folder
(named e.g., ``optimization``) with the following structure:

.. code-block:: bash

   optimization
   ├── run_example.py
   ├── template_simulation_script.py
   └── analysis_script.py

The optimization is started by executing:

.. code-block:: bash

   python run_example.py

The scripts needed to run this example can be seen below.

.. literalinclude:: ../../../examples/ionization_injection/run_example.py
   :language: python
   :caption: run_example.py (:download:`download <../../../examples/ionization_injection/run_example.py>`)

.. literalinclude:: ../../../examples/ionization_injection/template_simulation_script.py
   :language: python
   :caption: template_simulation_script.py (:download:`download <../../../examples/ionization_injection/template_simulation_script.py>`)

.. literalinclude:: ../../../examples/ionization_injection/analysis_script.py
   :language: python
   :caption: analysis_script.py (:download:`download <../../../examples/ionization_injection/analysis_script.py>`)


References
~~~~~~~~~~

.. [1] Sören Jalas, Manuel Kirchen, Philipp Messner, Paul Winkler, Lars Hübner,
   Julian Dirkwinkel, Matthias Schnepp, Remi Lehe, and Andreas R. Maier
   `Phys. Rev. Lett. 126, 104801 <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.126.104801>`_
   (2021)
