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

Running
~~~~~~~

Before using `optimas`, execute the following commands to activate
the virtual environment:

::

    module load cray-python/3.9.13.1
    source $HOME/sw/perlmutter/gpu/venvs/optimas/bin/activate
