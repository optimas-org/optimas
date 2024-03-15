Log an ``Exploration`` to Weights and Biases
============================================

`Weights and Biases <https://wandb.ai/site>`_ (W&B) is a powerful tool for
tracking and visualizing
machine learning experiments. Optimas has built-in support for logging to W&B,
allowing users to easily track and compare the performance of different
optimization runs.

This documentation provides a guide on how to use the
:class:`~optimas.loggers.WandBLogger` class
within Optimas to log an :class:`~optimas.explorations.Exploration`
to Weights and Biases.


Basic example
-------------

To log an :class:`~optimas.explorations.Exploration` to Weights and Biases,
you first need to instantiate
a :class:`~optimas.loggers.WandBLogger` object. This object requires several
parameters, including
your W&B API key, the project name, and optionally, a run name, run ID,
data types for specific parameters, and a user-defined function for
custom logs. For example:

.. code-block:: python

    from optimas.loggers import WandBLogger

    logger = WandBLogger(
        api_key="your_wandb_api_key",
        project="your_project_name",
        run="example_run",  # optional
    )

This logger can then be passed to an ``Exploration``, such as in the example
below:

.. code-block:: python

    from optimas.explorations import Exploration
    from optimas.generators import RandomSamplingGenerator
    from optimas.evaluators import FunctionEvaluator
    from optimas.loggers import WandBLogger
    from optimas.core import VaryingParameter, Objective


    # Define the function to be optimized
    def objective_function(inputs, outputs):
        x = inputs["x"]
        y = inputs["y"]
        outputs["result"] = x**2 + y**2


    # Define the evaluator
    evaluator = FunctionEvaluator(objective_function)

    # Define the generator
    generator = RandomSamplingGenerator(
        parameters=[
            VaryingParameter(name="x", lower_bound=-10, upper_bound=10),
            VaryingParameter(name="y", lower_bound=-10, upper_bound=10),
        ],
        objectives=[Objective(name="result", minimize=True)],
    )

    # Instantiate the WandBLogger
    logger = WandBLogger(
        api_key="your_wandb_api_key",
        project="your_project_name",
        run="example_run",
    )

    # Create the Exploration and pass the logger and evaluator
    exploration = Exploration(
        generator=generator, evaluator=evaluator, logger=logger
    )

    # Run the exploration
    exploration.run(n_evals=100)


Customizing the data type of the logger arguments
-------------------------------------------------

The ``data_types`` argument allows you to specify the W&B
`data type <https://docs.wandb.ai/ref/python/data-types/>`_ for specific
parameters when logging to Weights and Biases. This is useful for ensuring
that your data is logged in the desired format. The ``data_types`` should be
a dictionary where the keys are the names of the parameters you wish to
log, and the values are dictionaries containing the ``type`` and
``type_kwargs`` for each parameter.

For example, if you have defined two analyzed parameters called
``"parameter_1"`` and ``"parameter_2"`` that at each evaluation store
an image or matplotlib
figure and a numpy array, respectively, you can tell the logger to log the
first one as an image, and the second as a histogram:

.. code-block:: python

    data_types = {
        "parameter_1": {"type": wandb.Image, "type_kwargs": {}},
        "parameter_2": {"type": wandb.Histogram, "type_kwargs": {}},
    }

    logger = WandBLogger(
        api_key="your_wandb_api_key",
        project="your_project_name",
        data_types=data_types,
        # Other parameters...
    )


Defining custom logs
--------------------

By default, the ``WandBLogger`` will log the varying parameters, objectives
and analyzed parameters of the ``Exploration``.
If you want to include your own custom logs, you can provide a
``custom_logs`` function that generates them.
This function will be called every time a trial evaluation finishes.

The ``custom_logs`` function should take two arguments, which correspond to the
most
recently evaluated :class:`~optimas.core.Trial` and the currently active
``Generator``.
You do not need to use them, but they are there for convenience.
The function must then
return a dictionary with the appropriate shape to be given to ``wandb.log``.

Here's an example of how to define a ``custom_logs`` function:

.. code-block:: python

    def custom_logs(trial, generator):
        # Example: Log the best score so far
        best_score = None
        trials = generator.completed_trials
        for trial in trials:
            score = trial.data["result"]
            if best_score is None:
                best_score = score
            elif score < best_score:
                best_score = score
        return {"Best Score": best_score}


    logger = WandBLogger(
        api_key="your_wandb_api_key",
        project="your_project_name",
        custom_logs=custom_logs,
        # Other parameters...
    )
