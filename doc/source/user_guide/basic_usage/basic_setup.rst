Setting up an optimas run
=========================

Optimas is a library built on top of `libEnsemble <https://libensemble.readthedocs.io/>`_ that allows the execution of large parallel explorations (typically parameter scans or optimizations). The evaluations, which typically consist of resource-intensive simulations, can be carried out concurrently and with dedicated computational resources.

This section covers the basics components of optimas that are needed to set up a Python script for launching an :class:`~optimas.explorations.Exploration`.

Parameters to vary
~~~~~~~~~~~~~~~~~~
The first thing that needs to be specified is a list of parameters that should be varied during the optimization or scan. These are instances of :class:`~optimas.core.VaryingParameter`.

As an example, the code below shows how to define two parameters named ``x0`` and ``x1`` that can vary in the ranges [0, 15] and [-5, 5], respectively.

.. code-block:: python

    from optimas.core import VaryingParameter

    var_1 = VaryingParameter('x0', 0., 15.)
    var_2 = VaryingParameter('x1', -5., 5.)


Objectives and other analyzed parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Next, a list of objectives (:class:`~optimas.core.Objective`) should be provided. These are the quantities that are obtained from each evaluation and that the optimas will try to maximize or minimize (in case of an optimization) or simply to explore (in case of a parameter scan).

Optionally, a list of parameters (:class:`~optimas.core.Parameter`) that do not play a role in the optimization, but that should be analyzed at each evaluation (for example, because they provide useful information about the evaluations) can also be given.

The following code shows how to define one objective, named ``'f'``, that should be minimized and two diagnostics ``'diag_1'`` and ``'diag_2'`` that will also be calculated at each evaluation.

.. code-block:: python

    from optimas.core import Objective, Parameter

    obj = Objective('f', minimize=True)
    diag_1 = Parameter('diag_1')
    diag_2 = Parameter('diag_2')


Generator
~~~~~~~~~
The generator defines the strategy with which new points should be generated during the exploration. There are multiple generators implemented in optimas (see :ref:`generators`) that allow for various optimization strategies or parameter scans.

In the example below, the varying parameters, objectives and diagnostics defined in the previous sections are used to set up a generator for Bayesian optimization based on `Ax <https://ax.dev/>`_ that uses a single fidelity. ``n_init=4`` indicates that 4 random samples will be generated before the Bayesian optimization loop is started (see :class:`~optimas.generators.AxSingleFidelityGenerator` for more details).

.. code-block:: python

    from optimas.generators import AxSingleFidelityGenerator

    gen = AxSingleFidelityGenerator(
        varying_parameters=[var_1, var_2],
        objectives=[obj],
        analyzed_parameters=[diag_1, diag_2],
        n_init=4
    )


Evaluator
~~~~~~~~~
The evaluator is in charge of getting the trials suggested by the generator and evaluating them, returning the value of the objectives and other analyzed parameters.

There are two types of evaluators:

- :class:`~optimas.evaluators.FunctionEvaluator`: used to evaluate simple functions that do not demand large computational resources.
- :class:`~optimas.evaluators.TemplateEvaluator`: used to carry out expensive evaluations that are executed by running an external script. In this case, a template script should be given from which the scripts of each evaluation will be generated (see :ref:`simulation template` for more details). After executing the script, the evaluator analyzes the output of the evaluation with a user-provided function (see :ref:`analysis function` for more details).

The code below shows how to define a :class:`~optimas.evaluators.TemplateEvaluator` that executes a script generated from the template ``'template_simulation_script.py'`` and whose output is analyzed by a function ``analyze_simulation``. The script is executed with MPI, using by default a single process and no GPUs. This can be changed by specifying the ``n_procs`` and ``n_gpus`` attributes.

.. code-block:: python

    from optimas.evaluators import TemplateEvaluator

    ev = TemplateEvaluator(
        sim_template='template_simulation_script.py',
        analysis_func=analyze_simulation,
        # n_procs=2,
        # n_gpus=2
    )


Exploration
~~~~~~~~~~~
To create an :class:`~optimas.explorations.Exploration`, all that is needed is to indicate the generator and evaluator to use, as well as the maximum evaluations to perform and the number of simulation workers.

In the example below a maximim of 100 evaluations are carried out using 4 simulation workers. This means that up to 4 evaluation can be carried out in parallel at any time.

.. code-block:: python

    from optimas.explorations import Exploration

    exp = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=100,
        sim_workers=4
    )

To start the exploration, simply call ``exp.run()`` inside a ``if __name__ == '__main__':`` block such as

.. code-block:: python

    if __name__ == '__main__':
        exp.run()

This is needed in order to safely execute the parallel evaluations with some `mutliprocessing <https://docs.python.org/3/library/multiprocessing.html>`_ methods such as ``'spawn'`` (default on macOS).
