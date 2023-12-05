Grid sampling
=============


Description
~~~~~~~~~~~

This example shows how to perform a grid sampling parameter scan using a
:class:`~optimas.generators.GridSamplingGenerator` and a
:class:`~optimas.evaluators.TemplateEvaluator`.

The template simulation script evaluates a
simple function of two parameters :math:`x_0` and :math:`x_1`:

.. math::

   f(x_0, x_1) = -(x_0 + 10 \cos(x_0)) (x_1 + 5\cos(x_1))

and stores the outcome in a text file ``result.txt``. The ``analysis_func``
simply reads the value in this file.

You can adapt this example to your needs by replacing this
basic template with an actual simulation and writing the corresponding
analysis function. See see :ref:`optimas-with-simulations` for more details.

The :class:`~optimas.generators.GridSamplingGenerator` generates a uniform
multidimensional grid of samples to evaluate. The grid extends from the lower
to the upper bound of each :class:`~optimas.core.VaryingParameter` and is
divided in ``n_steps`` steps. In this case,
where :math:`l_b=0` and :math:`u_b=15`, the grid of sample looks like:


.. plot::
   :show-source-link: False


   import importlib.util

   spec = importlib.util.spec_from_file_location('run_example', '../../../examples/dummy_grid_sampling/run_example.py')
   module = importlib.util.module_from_spec(spec)
   spec.loader.exec_module(module)
   gen = module.gen

   all_trials = []
   while True:
       trial = gen.ask(1)
       if trial:
           all_trials.append(trial[0])
       else:
           break
   x0 = np.zeros(len(all_trials))
   x1 = np.zeros(len(all_trials))

   for i, trial in enumerate(all_trials):
       trial_params = trial.parameters_as_dict()
       x0[i] = trial_params['x0']
       x1[i] = trial_params['x1']

   fig, ax = plt.subplots()
   ax.scatter(x0, x1, s=3, label='generated evaluations')
   ax.set(
       xlabel=gen.varying_parameters[0].name,
       ylabel=gen.varying_parameters[1].name
   )
   ax.legend(loc='upper right')


Scripts
~~~~~~~

The two files needed to run this example should be located in the same folder
(named e.g., ``example``):

.. code-block:: bash

   example
   ├── run_example.py
   └── template_simulation_script.py

The example is executed by running

.. code-block:: bash

   python run_example.py

.. literalinclude:: ../../../examples/dummy_grid_sampling/run_example.py
   :language: python
   :caption: run_example.py (:download:`download <../../../examples/dummy_grid_sampling/run_example.py>`)

.. literalinclude:: ../../../examples/dummy_grid_sampling/template_simulation_script.py
   :language: python
   :caption: template_simulation_script.py (:download:`download <../../../examples/dummy_grid_sampling/template_simulation_script.py>`)
