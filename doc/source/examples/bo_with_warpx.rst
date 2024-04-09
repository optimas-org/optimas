.. _bo-with-warpx:

Optimization with WarpX
=======================


Description
~~~~~~~~~~~

This examples shows how to perform a Bayesian optimization of a laser-plasma
accelerator (LPA) using WarpX simulations.

The LPA to be optimized is based on 

The objective function to optimize (maximize) is defined as

.. math::
   f = \epsilon_f + 100\epsilon_i \left(1.0 - \frac{Q_f}{Q_i}\right)


where :math:`\epsilon_i` and :math:`\epsilon_f` are the initial and final beam emittances, respectively, 
and :math:`Q_i` and :math:`Q_f` are the initial and final beam charges.
This objective is optimized by tuning 2 parameters:

- ``'adjust_factor'``: parameter in the range :math:`[0.7, 1.05]` that scales the 
  strength of the magnetic field between the first and second stage.
  The value ``adjust_factor=1`` corresponds to a focusing strength of :math:`454535.7\, \mathrm{T/m}`.
- ``'zlen'``: the left or starting position position of the laser in millimetres, with range
  :math:`[0.32, 0.347]`.

The optimization is carried out using an
:class:`~optimas.generators.AxSingleFidelityGenerator` and a
:class:`~optimas.evaluators.TemplateEvaluator`. In this case, the function
``analyze_simulation`` that analyzes the output of each simulation is defined
in a separate file ``analysis_script.py`` and imported into the main
optimas script.

The example is set up to make use of a system of 4 GPUs, where each WarpX
simulation uses a single GPU and 4 simulations are carried out in parallel.


Scripts
~~~~~~~

The files needed to run the optimization should be located in a folder
(named e.g., ``optimization``) with the following structure:

.. code-block:: bash

   optimization
   ├── run_example.py
   ├── template_simulation_script
   ├── analysis_script.py
   └── warpx

Note that the ``WarpX`` RZ executable ``warpx`` needs to be in the ``optimization`` folder.
The optimization is started by executing:

.. code-block:: bash

   python run_example.py

The scripts needed to run this example can be seen below.

.. literalinclude:: ../../../examples/multi_stage/run_example.py
   :language: python
   :caption: run_example.py (:download:`download <../../../examples/multi_stage/run_example.py>`)

.. literalinclude:: ../../../examples/multi_stage/template_simulation_script
   :language: bash
   :caption: template_simulation_script (:download:`download <../../../examples/multi_stage/template_simulation_script>`)

.. literalinclude:: ../../../examples/multi_stage/analysis_script.py
   :language: python
   :caption: analysis_script.py (:download:`download <../../../examples/multi_stage/analysis_script.py>`)
