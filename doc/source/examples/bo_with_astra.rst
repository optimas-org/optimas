.. _bo-with-astra:

Optimization with ASTRA
==========================


Description
~~~~~~~~~~~

This examples shows how to perform a multi-objective Bayesian optimization of
beam parameters using `ASTRA <https://www.desy.de/~mpyflo/>`_.

The setup is based on a beamline example from the ASTRA manual which can be found
`here <https://www.desy.de/~mpyflo/EXAMPLES/Manual_Example/>`_.

Two optimization parameters are used:
- the RF phase of the cavity ``'RF_phase'``, which is varied in the range :math:`[-2.5, 2.5]`,
- and the solenoid strength ``'B_sol'``which is varied in the range :math:`[0.12, 0.38]`.

Two beam parameters are minimized:
- the bunch_length,
- and the transverse emittances ``'emittance'`` in :math:`\mathrm{µm}`, which are combined into one single parameter: :math:`\log{em_n_x * em_n_y}` and where the logarithm is used for better optimization.

In addition, the transverse normalized emittances in :math:`x` and :math:`y` are stored as additional analyzed parameters ``'emittance_x'`` and ``'emittance_y'``.

The optimization is carried out using an
:class:`~optimas.generators.AxSingleFidelityGenerator` and a
:class:`~optimas.evaluators.TemplateEvaluator`. In this case, the function
``analyze_simulation`` that analyzes the output of each simulation is defined
in a separate file ``analysis_script.py`` and imported into the main
optimas script.

The ASTRA simulation template ``ASTRA_example.in`` requires additional files. These can be downloaded from the ASTRA `website <https://www.desy.de/~mpyflo/EXAMPLES/Manual_Example/>`_ and are the input particle distribution ``Example.ini``, the RF field profile ``3_cell_L-Band.dat``, and the solenoid field profile ``Solenoid.dat``.
These files need to be passed to the ``TemplateEvaluator`` using the ``sim_files`` argument.

The path to the ASTRA executable needs to be specified in the ``TemplateEvaluator`` using the ``executable`` argument.

Scripts
~~~~~~~

The files needed to run the optimization should be located in a folder
(named e.g., ``optimization``) with the following structure:

.. code-block:: bash

   optimization
   ├── run_optimization_serial_ASTRA.py
   ├── ASTRA_example.in
   └── analysis_script.py
   └── Example.ini
   └── 3_cell_L-Band.dat
   └── Solenoid.dat

The optimization is started by executing:

.. code-block:: bash

   python run_optimization_serial_ASTRA.py

The main scripts needed to run this example can be seen below.

.. literalinclude:: ../../../examples/astra/run_optimization_serial_ASTRA.py
   :language: python
   :caption: run_optimization_serial_ASTRA.py (:download:`download <../../../examples/astra/run_optimization_serial_ASTRA.py>`)

.. literalinclude:: ../../../examples/astra/ASTRA_example.in
   :language: fortran
   :caption: ASTRA_example.in (:download:`download <../../../examples/astra/ASTRA_example.in>`)

.. literalinclude:: ../../../examples/astra/analysis_script.py
   :language: python
   :caption: analysis_script.py (:download:`download <../../../examples/astra/analysis_script.py>`)
