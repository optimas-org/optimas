Installation on a local computer
--------------------------------

The recommended approach is to install Optimas in a ``conda`` environment.

Install basic dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code::

    conda install numpy pandas

Install PyTorch
~~~~~~~~~~~~~~~

If your computer does not feature a CUDA-capable GPU, install PyTorch for CPU:

.. code::

    conda install pytorch cpuonly -c pytorch


If you have a CUDA-capable GPU and want to take make it available to Optimas,
install PyTorch with:

.. code::

    conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia

Install ``mpi4py``
~~~~~~~~~~~~~~~~~~
If your system has already an MPI implementation installed, install ``mpi4py``
using ``pip``:

.. code::

    pip install mpi4py

This will make sure that Optimas uses the existing MPI. The recommended
MPI implementation is MPICH.

If you don't have an existing MPI installation, the recommended approach is to
install ``mpi4py`` from ``conda``, including the MPI implementation corresponding
to your operating system.

On Linux and macOS:

.. code::

    conda install -c conda-forge mpi4py mpich

On Windows:

.. code::

    conda install -c conda-forge mpi4py msmpi

Install Optimas
~~~~~~~~~~~~~~~
Install the latest release from PyPI

.. code::

    pip install optimas
