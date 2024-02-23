Installation on Maxwell (DESY)
------------------------------

Log into a Maxwell display node

.. code::

    ssh <username>@max-display.desy.de


Create a ``profile.optimas`` file in your home directory with the following content

.. code::

    source /etc/profile.d/modules.sh
    module purge
    module load intel.2020
    module load maxwell cuda/11.8
    module load mpi/mpich-3.2-x86_64
    module load maxwell mamba
    . mamba-init

you can do this from the command line by executing ``cat > ~/profile.optimas`` and
pasting the lines above. To save the file, finalize by pressing ``Ctrl+D``.

Load the source file

.. code::

    source ~/profile.optimas


Create a new environment for optimas

.. code::

    mamba create -n optimas_env python=3.11


Activate the environment

.. code::

    mamba activate optimas_env


Install ``mpi4py``

.. code::

    pip install mpi4py --no-cache-dir


Install ``optimas`` with all dependencies if you plan to do Bayesian optimization
(see :ref:`dependencies` for more details).

.. code::

    pip install 'optimas[all]'


Installing FBPIC and Wake-T (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A common use case for ``optimas`` is to optimize the output of a plasma acceleration
simulation with FBPIC or Wake-T (or both). If you need any of these tools, you
can follow the instructions below to install them in your ``optimas_env``
environment.

Install FBPIC:

.. code::

    mamba install cudatoolkit=11.8
    pip install cupy-cuda11x
    pip install fbpic


Install Wake-T:

.. code::

    pip install Wake-T

Install openPMD-viewer for data analysis

.. code::

    pip install openPMD-viewer


Running an optimas job
~~~~~~~~~~~~~~~~~~~~~~
The following script can be used to submit an ``optimas`` job to Maxwell.
This example sends the job to the ``maxgpu`` queue. It assumes that you
need one node with 4 GPUs (NVIDIA A100) and that your optimas script is called
``run_optimization.py``.

.. code::

    #!/bin/bash

    #SBATCH --partition=maxgpu
    #SBATCH --time=24:00:00
    #SBATCH --nodes=1
    #SBATCH --constraint="A100&GPUx4"
    #SBATCH --job-name  optimas
    #SBATCH --output    stdout
    #SBATCH --error     stderr
    #SBATCH --mail-type BEGIN,END

    export SLURM_EXACT=1

    # Activate environment
    source ~/profile.optimas
    mamba activate optimas_env

    python run_optimization.py
