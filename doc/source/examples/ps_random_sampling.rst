Random sampling
===============


Description
~~~~~~~~~~~

This example shows how to perform a random parameter scan using a
:class:`~optimas.generators.RandomSamplingGenerator` and a
:class:`~optimas.evaluators.TemplateEvaluator`.

The template simulation script evaluates a
simple function of two parameters :math:`x_0` and :math:`x_1`:

.. math::

   f(x_0, x_1) = -(x_0 + 10 \cos(x_0)) (x_1 + 5\cos(x_1))

and stores the outcome in a text file ``result.txt``. The ``analysis_func``
simply reads the value in this file.

You can adapt this example to your needs by replacing this
basic template with an actual simulation and writing the corresponding
analysis function. See see :ref:`simulation template` and
:ref:`analysis function` for more details.

The :class:`~optimas.generators.RandomSamplingGenerator` draws samples from
a ``'normal'`` distribution that, for each parameter, is centered at
:math:`c = l_b - u_b` with standard deviation :math:`\\sigma = u_b - c`, 
where :math:`l_b` and :math:`u_b` are, respectively, the lower and upper
bounds of the parameter. Other distributions are also available. In this case,
where :math:`l_b=0` and :math:`u_b=15`, the drawn samples result in a
distribution such as:


.. plot::
   :show-source-link: False

   import importlib.util
   from matplotlib.patches import Ellipse

   spec = importlib.util.spec_from_file_location('run_example', '../../../examples/dummy_random/run_example.py')
   module = importlib.util.module_from_spec(spec)
   spec.loader.exec_module(module)
   gen = module.gen

   all_trials = []
   while len(all_trials) <= 100:
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
   ax.scatter(gen._center[0], gen._center[1], s=10, color='tab:red', label='center', marker='x')
   ellipse = Ellipse((gen._center[0], gen._center[1]),
        width=gen._width[0] * 2,
        height=gen._width[1] * 2,
        facecolor='none',
        edgecolor='tab:red',
        label='standard deviation')
   ax.add_patch(ellipse)
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

You can find both example scripts below.

.. literalinclude:: ../../../examples/dummy_random/run_example.py
   :language: python
   :caption: run_example.py

.. literalinclude:: ../../../examples/dummy_random/template_simulation_script.py
   :language: python
   :caption: template_simulation_script.py