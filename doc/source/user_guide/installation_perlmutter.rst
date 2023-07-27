Installation on Perlmutter (NERSC)
----------------------------------

Installation
~~~~~~~~~~~~

Execute the following commands in order to create a dedicated Python
environment, in which to install `optimas`.

::

    module load cray-python/3.9.13.1

    python3 -m pip install --user --upgrade pip
    python3 -m pip install --user virtualenv
    python3 -m pip cache purge
    python3 -m venv $HOME/sw/perlmutter/gpu/venvs/optimas
    source $HOME/sw/perlmutter/gpu/venvs/optimas/bin/activate

    python3 -m pip install optimas

Running an optimas job
~~~~~~~~~~~~~~~~~~~~~~

In order to run a new optimas job, create a new folder in the ``$SCRATCH``
directory, and add the files needed to describe your optimization, i.e. typically:

* a file ``run_optimization.py``
* a template simulation input script
* a file ``analysis_script.py`` that postprocesses the simulation results

(See a typical example `here <https://github.com/optimas-org/optimas/tree/main/examples/ionization_injection>`_.)

Then, create a file ``submission_script`` with the following content:

::

    #!/bin/bash -l
    #SBATCH -t <walltime>
    #SBATCH -N <n_nodes>
    #SBATCH -A <nersc_account>
    #SBATCH -C gpu
    #SBATCH -q regular
    #SBATCH --exclusive
    #SBATCH --gpu-bind=none
    #SBATCH --gpus-per-node=4

    module load cray-python/3.9.13.1
    source $HOME/sw/perlmutter/gpu/venvs/optimas/bin/activate

    python run_optimization.py

where ``<walltime>``, ``<n_nodes>`` and ``<nersc_account>`` should be replaced
by the wall time, number of nodes and NERSC account number that you want to use.

Then run:

::

    sbatch submission_script