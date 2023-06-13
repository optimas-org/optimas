.. _simulation template:

Creating a simulation template
==============================
.. note::

   If your simulation script requires additional files (e.g., datasets that
   will be read by the simulations), indicate this to the ``TemplateEvaluator``
   by passing the list of files to the argument ``sim_files=['/path/to/file_1', '/path/to/file_2']``.
   These files will be copied to the simulation directory.

.. note::

   The template script should be should be a Python script or 