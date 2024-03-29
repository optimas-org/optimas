.. _optimas-with-simulations:

Running simulations
===================

A common use case for optimas is to run an optimization or parameter scan
where each evaluation consists of running a simulation that is defined in an
external file. This workflow requires:

- A template of the simulation script that indicates where the values
  of the :class:`~optimas.core.VaryingParameter`\s should be placed.
- A function to analyze the simulation output and determine the value of the
  :class:`~optimas.core.Objective`\s and other parameters.

This is all handled by defining a
:class:`~optimas.evaluators.TemplateEvaluator` that, in it's most basic form,
would look something like

.. code-block:: python

    from optimas.evaluators import TemplateEvaluator


    def analyze_simulation(simulation_directory, output_params):
        pass


    ev = TemplateEvaluator(
        sim_template="template_simulation_script.py",
        analysis_func=analyze_simulation,
    )


Creating a simulation template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The scripts for each simulation are created from a template where the
values of the :class:`~optimas.core.VaryingParameter`\s
are introduced using `Jinja <https://jinja.palletsprojects.com>`_ syntax, that
is, using the double-bracket notation ``{{var_name}}``, where ``var_name`` is
the name of the :class:`~optimas.core.VaryingParameter`.

As a basic example, a template for a Python script that takes in two
:class:`~optimas.core.VaryingParameter`\s called ``'x'`` and ``'y'``,
computes ``x + y``, and stores the result in a text file would look like:

.. code-block:: python
   :caption: template_simulation_script.py

   result = {{x}} + {{y}}

   with open("result.txt", "w") as f:
       f.write("%f" % result)


To see a more elaborate template script that actually launches a simulation,
you can check out the example :ref:`bo-with-fbpic`.


Defining an analysis function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``analysis_func`` given to the
:class:`~optimas.evaluators.TemplateEvaluator` should be a user-defined
function that accepts two arguments: a string with the path of the
simulation directory and a dictionary where the value of the output parameters
(e.g., the objectives) will be stored.
You can define this function directly in the main optimas script, or import it
from another file.

As an example, assuming that the result of ``x + y`` in the previous section
is an :class:`~optimas.core.Objective` called ``'f'``, the analysis function
would look like:

.. code-block:: python

   def analyze_simulation(simulation_directory, output_params):
       """Analyze the simulation output.

       This method analyzes the output generated by the simulation to
       obtain the value of the optimization objective and other analyzed
       parameters, if specified. The value of these parameters has to be
       given to the `output_params` dictionary.

       Parameters
       ----------
       simulation_directory : str
          Path to the simulation folder where the output was generated.
       output_params : dict
          Dictionary where the value of the objectives and analyzed parameters
          will be stored. There is one entry per parameter, where the key
          is the name of the parameter given by the user.

       Returns
       -------
       dict
          The `output_params` dictionary with the results from the analysis.
       """
       # Read back result from file
       with open("result.txt") as f:
           result = float(f.read())
       # Fill in output parameters.
       output_params["f"] = result
       return output_params


Assigning computational resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optimas executes the simulations using MPI with the amount of resources
(number of MPI processes and GPUs) specified by the
``n_procs`` and ``n_gpus`` attributes of the
:class:`~optimas.evaluators.TemplateEvaluator`. By default:

- If no ``n_procs`` nor ``n_gpus`` are given, the simulations are run using a
  single MPI process and no GPUs.
- If only ``n_gpus`` is given, then ``n_procs=n_gpus``.

For example, running a simulation with 2 GPUs and one MPI process per GPU
would be done with

.. code-block:: python
   :emphasize-lines: 4

   ev = TemplateEvaluator(
       sim_template="template_simulation_script.py",
       analysis_func=analyze_simulation,
       n_gpus=2,
   )


Including additional simulation files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your simulations require additional files (e.g., datasets that
will be loaded by the simulation script), indicate this to the
:class:`~optimas.evaluators.TemplateEvaluator`
by passing the list of files to the argument ``sim_files``.
These files will be copied to the simulation directory together with the
simulation script.

.. code-block:: python
   :emphasize-lines: 4

   ev = TemplateEvaluator(
       sim_template="template_simulation_script.py",
       analysis_func=analyze_simulation,
       sim_files=["/path/to/file_1", "/path/to/file_2"],
   )


Executing a non-Python simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your template is a not a Python script, make sure to specify the name or
path to the ``executable`` that will run your simulation.

.. code-block:: python
   :emphasize-lines: 3

   ev = TemplateEvaluator(
       sim_template="template_simulation_script.txt",
       executable="/path/to/my_executable",
       analysis_func=analyze_simulation,
   )


Using a custom environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``env_script`` and ``env_mpi`` parameters allow you to customize the
environment in which your simulation runs.

``env_script`` takes the path to a shell script that sets up the
environment by loading the necessary dependencies, setting environment
variables, or performing other setup tasks required by your simulation.

This script will look different depending on your system and use
case, but it will typically be something like

.. code-block:: bash

    #!/bin/bash

    # Set environment variables
    export VAR1=value1
    export VAR2=value2

    # Load a module
    module load module_name


If the script loads a different MPI version than the one in the ``optimas``
environment, make sure to specify the loaded version with the ``env_mpi``
argument. For example:

.. code-block:: python
   :emphasize-lines: 5,6

   ev = TemplateEvaluator(
       sim_template="template_simulation_script.txt",
       executable="/path/to/my_executable",
       analysis_func=analyze_simulation,
       env_script="/path/to/my_env_script.sh",
       env_mpi="openmpi",
   )


See :class:`~optimas.evaluators.TemplateEvaluator` for more details.


Running a chain of simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~optimas.evaluators.ChainEvaluator` is designed for use cases
where each evaluation involves several steps, each step being a simulation
with a different simulation code.

The steps are defined by a list of ``TemplateEvaluators`` ordered in the
sequence in which they should be executed. Each step can request a different
number of resources, and the ``ChainEvaluator`` gets allocated the maximum
number of processes (``n_procs``) and GPUs (``n_gpus``) that every step might
request.
For instance, if one step requires ``n_procs=20`` and ``n_gpus=0``, and a
second step requires ``n_procs=4`` and ``n_gpus=4``, each evaluation will
get assigned ``n_procs=20`` and ``n_gpus=4``. Then each step will only
make use of the subset of resources it needs.

Here is a basic example of how to use ``ChainEvaluator``:

.. code-block:: python

    from optimas.evaluators import TemplateEvaluator, ChainEvaluator

    # define your TemplateEvaluators
    ev1 = TemplateEvaluator(
        sim_template="template_simulation_script_1.py",
        analysis_func=analyze_simulation_1,
    )

    ev2 = TemplateEvaluator(
        sim_template="template_simulation_script_2.py",
        analysis_func=analyze_simulation_2,
    )

    # use them in ChainEvaluator
    chain_ev = ChainEvaluator([ev1, ev2])


In this example, ``template_simulation_script_1.py`` and
``template_simulation_script_2.py`` are your simulation scripts for the
first and second steps, respectively. ``analyze_simulation_1`` and
``analyze_simulation_2`` are functions that analyze the output of each
simulation. There is no need to provide an analysis function for every step,
but at least one should be defined.
