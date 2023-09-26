Setting up an optimas run
=========================

This section covers the basic workflow of setting up an optimas
:class:`~optimas.explorations.Exploration`, which is typically used to launch
an optimization or parameter scan. This involves:

- Specifying the parameters that should be varied during the exploration.
- Specifying the optimization objectives and other parameters that should
  analyzed for each evaluation.
- Choosing a generator. This determines the strategy with which new evaluations
  are generated.
- Choosing an evaluator. This determines how the evaluations are performed and
  which computational resources are assigned to them.
- Specifying how many evaluations should be carried out in parallel and the
  criteria for ending the exploration.


Parameters to vary
~~~~~~~~~~~~~~~~~~
The parameters to vary (:class:`~optimas.core.VaryingParameter`) are the
parameters that should be tuned or scanned during the exploration.
For example, if we want to see how the outcome of an evaluation depends on two
parameters named ``x0`` and ``x1`` that can vary in the ranges [0, 15] and
[-5, 5], we would define them as

.. code-block:: python

    from optimas.core import VaryingParameter

    var_1 = VaryingParameter("x0", 0.0, 15.0)
    var_2 = VaryingParameter("x1", -5.0, 5.0)


Objectives and other analyzed parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The objectives (:class:`~optimas.core.Objective`) define the outcomes of an
evaluation that optimas should optimize (maximize or minimize) or scan.

Optionally, a list of parameters (:class:`~optimas.core.Parameter`) that do not
play a role in the optimization but that should be analyzed at each evaluation
(for example, because they provide useful information about the evaluations)
can also be given.

The following code shows how to define one objective ``'f'`` that
should be minimized and two diagnostics ``'diag_1'`` and ``'diag_2'`` that will
also be calculated for each evaluation.

.. code-block:: python

    from optimas.core import Objective, Parameter

    obj = Objective("f", minimize=True)
    diag_1 = Parameter("diag_1")
    diag_2 = Parameter("diag_2")


Generator
~~~~~~~~~
The generator defines the strategy with which new points are generated
during the exploration. There are multiple generators implemented in optimas
(see :ref:`generators`) that allow for various optimization strategies or
parameter scans.

In the example below, the varying parameters, objectives and diagnostics
defined in the previous sections are used to set up a single-fidelity Bayesian
optimizer based on `Ax <https://ax.dev/>`_.
``n_init=4`` indicates that 4 random samples will be generated before the
Bayesian optimization loop is started (see
:class:`~optimas.generators.AxSingleFidelityGenerator` for more details).

.. code-block:: python

    from optimas.generators import AxSingleFidelityGenerator

    gen = AxSingleFidelityGenerator(
        varying_parameters=[var_1, var_2],
        objectives=[obj],
        analyzed_parameters=[diag_1, diag_2],
        n_init=4,
    )


Evaluator
~~~~~~~~~
The evaluator is in charge of getting the trials suggested by the generator and
evaluating them, returning the value of the objectives and other analyzed
parameters.

There are two types of evaluators:

- :class:`~optimas.evaluators.FunctionEvaluator`: used to evaluate Python
  functions that do not demand large computational resources. Each evaluation
  will be carried out in a different process using either
  `multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`_
  or MPI.
- :class:`~optimas.evaluators.TemplateEvaluator`: used to carry out expensive
  evaluations that are executed by running an external script. In this case, a
  template script should be given from which the scripts of each evaluation
  will be generated.
  Each evaluation is executed using MPI with the amount or resources (number of
  processes and GPUs) specified by the user. After executing the script, the
  output of the evaluation is analyzed with a user-defined function that
  calculates the value of the objectives and other analyzed parameters.
  See :ref:`optimas-with-simulations` for more details about how to use a
  :class:`~optimas.evaluators.TemplateEvaluator`.

The code below shows an example of how to define a
:class:`~optimas.evaluators.TemplateEvaluator` that executes a script generated
from the template ``'template_simulation_script.py'`` and whose output is
analyzed by a function ``analyze_simulation``. The script is executed with MPI,
using by default a single process and no GPUs. This can be
changed by specifying the ``n_procs`` and ``n_gpus`` attributes.

.. code-block:: python

    from optimas.evaluators import TemplateEvaluator

    ev = TemplateEvaluator(
        sim_template="template_simulation_script.py",
        analysis_func=analyze_simulation,
        # n_procs=2,
        # n_gpus=2
    )


Exploration
~~~~~~~~~~~
The :class:`~optimas.explorations.Exploration` is the main class that
coordinates the generation and execution of evaluations. In addition to
the generator and evaluator to use, it requires the user to specify the maximum
number evaluations to perform and the number of simulation workers.

In the example below, a maximum of 100 evaluations will be carried out using 4
simulation workers. This means that up to 4 evaluation will be performed in
parallel at any time.

.. code-block:: python

    from optimas.explorations import Exploration

    exp = Exploration(generator=gen, evaluator=ev, max_evals=100, sim_workers=4)

The exploration is started by executing ``exp.run()`` inside a
``if __name__ == '__main__':`` block:

.. code-block:: python

    if __name__ == "__main__":
        exp.run()

This is needed in order to safely execute the exploration in systems using the
``'spawn'``
`multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`_
method (default on macOS).
