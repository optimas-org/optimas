.. _bo-with-hipace:

Optimization with HiPACE++
==========================


Description
~~~~~~~~~~~

This examples shows how to perform a Bayesian optimization of a PWFA using
HiPACE++.

The setup is a simple driver-witness configuration where the witness is
optimized to maximize the objetive

.. math::

   f = \frac{\sqrt{Q} E_{MED}}{100 E_{MAD}}


where :math:`Q` is the beam charge, :math:`E_{MED}` is the median energy, and
:math:`E_{MAD}` is the median absolute deviation energy spread. The only
optimization parameter is the charge:

- ``'witness_charge'``: parameter in the range :math:`[0.05, 1.]` in units of
  :math:`\mathrm{nC}`.

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

.. literalinclude:: ../../../examples/hipace/run_example.py
   :language: python
   :caption: run_example.py (:download:`download <../../../examples/hipace/run_example.py>`)

.. literalinclude:: ../../../examples/hipace/template_simulation_script
   :language: python
   :caption: template_simulation_script.py (:download:`download <../../../examples/hipace/template_simulation_script>`)

.. literalinclude:: ../../../examples/hipace/analysis_script.py
   :language: python
   :caption: analysis_script.py (:download:`download <../../../examples/hipace/analysis_script.py>`)
