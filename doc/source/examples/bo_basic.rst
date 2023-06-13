Basic optimization with simulations
===================================


Description
~~~~~~~~~~~

This example illustrates how to run a generic Bayesian optimization with
simulations. This typically requires:

- An `optimas` script for defining and running the optimization.
- A template simulation script.
- A function to analyze the simulation output.

In this generic example, the "simulations" will be simple evaluations of an
analytical equation. For a real use case, the simple template that evaluates
this expression can be replaced by an actual simulation script.

.. note::

   If you want to adapt this example to a case where the simulation template
   is not a Python script, make sure to pass ``executable=<my_executable>``
   as an argument to the ``TemplateEvaluator``, where ``<my_executable>`` is
   the path to the executable that will run your simulation script.

   For additional details about how to set up an template simulation script see
   :ref:`simulation template`.


Scripts
~~~~~~~

The two files needed to run the optimization should be located in a folder
(named e.g., ``optimization``) with the following structure:

.. code-block:: bash

   optimization
   ├── run_example.py
   └── template_simulation_script.py

The optimization is started by executing:

.. code-block:: bash

   python run_example.py

You can find both example scripts below.


.. literalinclude:: ../../../examples/dummy/run_example.py
   :language: python
   :caption: run_example.py (:download:`download <../../../examples/dummy/run_example.py>`)

.. literalinclude:: ../../../examples/dummy/template_simulation_script.py
   :language: python
   :caption: template_simulation_script.py  (:download:`download <../../../examples/dummy/template_simulation_script.py>`)
