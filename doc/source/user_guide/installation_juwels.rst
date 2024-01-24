Installation on JUWELS Booster (JSC)
------------------------------------

Log into a JUWELS booster node either with ``ssh`` or by opening a terminal
in the `JupyterLabs <https://jupyter-jsc.fz-juelich.de/>`_ (choose JUWELS as
system and LoginNodeBooster as Partition).


Create a ``profile.optimas`` file in your home directory with the following content

.. code::

    module purge
    module load Stages/2023
    module load GCC
    module load ParaStationMPI
    module load CUDA
    module load SciPy-bundle
    module load mpi4py

you can do this from the command line by executing ``cat > ~/profile.optimas`` and
pasting the lines above. To save the file, finalize by pressing ``Ctrl+D``.

Load the source file

.. code::

    source ~/profile.optimas


Create a new environment for optimas

.. code::

    python -m venv $PROJECT/<username>/pyenvs/optimas_env


Activate the environment

.. code::

    source $PROJECT/<username>/pyenvs/optimas_env/bin/activate


Install ``optimas`` with all dependencies if you plan to do Bayesian optimization
(see :ref:`dependencies` for more details).

.. code::

    pip install optimas[all]


Installing FBPIC and Wake-T (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A common use case for ``optimas`` is to optimize the output of a plasma acceleration
simulation with FBPIC or Wake-T (or both). If you need any of these tools, you
can follow the instructions below to install them in your ``optimas_env``
environment.

Install FBPIC:

.. code::

    pip install cupy-cuda117
    pip install fbpic


Install Wake-T:

.. code::

    pip install Wake-T

Install openPMD-viewer for data analysis

.. code::

    pip install openPMD-viewer


Running an optimas job
~~~~~~~~~~~~~~~~~~~~~~
The following script can be used to submit an ``optimas`` job to the JUWELS
booster (i.e., the ``booster`` partition). For testing, use the
``develbooster`` partition instead. This script assumes that you
need one node with 4 GPUs and that your optimas script is called
``run_optimization.py``.

Make sure to replace ``project_account``, ``user_email`` and ``username`` with
your data.

.. code::

    #!/bin/bash

    #SBATCH --job-name optimas
    #SBATCH --nodes=1
    #SBATCH --partition=booster
    #SBATCH --account=<project_account>
    #SBATCH --time=12:00:00
    #SBATCH --output=stdout
    #SBATCH --error=stderr
    #SBATCH --mail-type=END
    #SBATCH --mail-user=<user_email>

    # Activate environment
    source ~/profile.optimas
    source $PROJECT/<username>/pyenvs/optimas_env/bin/activate

    python run_optimization.py
