Exploration setup
=================

This section covers the basic workflow of setting up an optimas
:class:`~optimas.explorations.Exploration`, which is typically used to launch
an optimization or parameter scan. This involves:

- Specifying the variables, objectives, and observables via a
  :class:`~gest_api.vocs.VOCS` object.
- Choosing a generator. This determines the strategy with which new evaluations
  are generated.
- Choosing an evaluator. This determines how the evaluations are performed and
  which computational resources are assigned to them.
- Specifying how many evaluations should be carried out in parallel and the
  criteria for ending the exploration.


VOCS
~~~~
The variables, objectives, constraints, and observables of an exploration are
all specified through a single :class:`~gest_api.vocs.VOCS` object.

- **Variables** are the parameters that should be tuned or scanned during the
  exploration, together with their allowed range.
- **Objectives** define the outcomes of an evaluation that optimas should
  optimize (maximize or minimize).
- **Observables** are optional quantities that do not play a role in the
  optimization but that should be recorded at each evaluation (for example,
  because they provide useful diagnostic information).

For example, if we want to optimize an objective ``'f'`` (to be minimized)
over two parameters ``x0`` and ``x1`` in the ranges [0, 15] and [-5, 5],
while also recording two diagnostics ``'diag_1'`` and ``'diag_2'``, we would
define the VOCS as:

.. code-block:: python

    from gest_api.vocs import VOCS

    vocs = VOCS(
        variables={
            "x0": [0.0, 15.0],
            "x1": [-5.0, 5.0],
        },
        objectives={"f": "MINIMIZE"},
        observables=["diag_1", "diag_2"],
    )


Generator
~~~~~~~~~
The generator defines the strategy with which new points are generated
during the exploration. There are multiple generators implemented in optimas
(see :ref:`generators`) that allow for various optimization strategies or
parameter scans.

In the example below, the VOCS defined above is used to set up a
single-fidelity Bayesian optimizer based on `Ax <https://ax.dev/>`_.
``n_init=4`` indicates that 4 random samples will be generated before the
Bayesian optimization loop is started (see
:class:`~optimas.generators.AxSingleFidelityGenerator` for more details).

.. code-block:: python

    from optimas.generators import AxSingleFidelityGenerator

    gen = AxSingleFidelityGenerator(vocs=vocs, n_init=4)


Using an external generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you have a generator from a third-party library that follows the
`gest-api <https://github.com/campa-consortium/gest-api>`_ generator standard,
you can integrate it with Optimas using
:class:`~optimas.generators.ExternalGenerator`. The external generator must be
instantiated and configured first, then passed to ``ExternalGenerator`` as a
wrapper. The external library itself must be installed separately (see
:ref:`dependencies`).

Known libraries containing generators compatible with this interface include
`Xopt <https://github.com/xopt-org/Xopt>`_ and `libEnsemble
<https://github.com/Libensemble/libensemble>`_.

Using a generic ``gest-api``-compatible generator:

.. code-block:: python

    from optimas.generators import ExternalGenerator
    from gest_api.vocs import VOCS
    from some_library import SomeGenerator

    vocs = VOCS(
        variables={"x0": [0.0, 15.0], "x1": [-5.0, 5.0]},
        objectives={"f": "MINIMIZE"},
    )

    ext_gen = SomeGenerator(vocs=vocs)
    gen = ExternalGenerator(ext_gen=ext_gen, vocs=vocs)

Using an `Xopt <https://github.com/xopt-org/Xopt>`_ generator specifically:

.. code-block:: python

    from optimas.generators import ExternalGenerator
    from gest_api.vocs import VOCS
    from xopt.generators.bayesian.expected_improvement import (
        ExpectedImprovementGenerator,
    )

    vocs = VOCS(
        variables={"x0": [0.0, 15.0], "x1": [-5.0, 5.0]},
        objectives={"f": "MINIMIZE"},
    )

    # Create and (optionally) pre-seed the external generator.
    ext_gen = ExpectedImprovementGenerator(vocs=vocs)
    ext_gen.ingest([{"x0": 1.0, "x1": 0.5, "f": 3.2}])

    # Wrap it for use with optimas.
    gen = ExternalGenerator(ext_gen=ext_gen, vocs=vocs)


Evaluator
~~~~~~~~~
The evaluator is in charge of getting the trials suggested by the generator and
evaluating them, returning the value of the objectives and other observables.

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
  calculates the value of the objectives and other observables.
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
