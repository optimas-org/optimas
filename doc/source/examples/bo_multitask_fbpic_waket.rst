Multitask optimization with FBPIC and Wake-T
============================================

Description
~~~~~~~~~~~

This is an advanced example that shows how perform a multitask Bayesian
optimization using two simulations codes of different fidelity
(FBPIC and Wake-T). The scripts provided here can be used to reproduce
the results from the paper

- "Bayesian optimization of laser-plasma accelerators
  assisted by reduced physical models" by A. Ferran Pousa, S. Jalas, M. Kirchen,
  A. Martinez de la Ossa, M. Thévenet, J. Larson, S. Hudson, A. Huebl, J.-L. Vay,
  and R. Lehe (`link <https://arxiv.org/abs/2212.12551>`_).


Requirements
~~~~~~~~~~~~
In addition to optimas, the following packages should be installed:

- `FBPIC <https://fbpic.github.io/>`_
- `Wake-T <https://wake-t.readthedocs.io/>`_
- `VisualPIC <https://github.com/AngelFP/VisualPIC>`_


Scripts
~~~~~~~

Files included:

- :download:`run_opt.py <../../../examples/multitask_lpa_fbpic_waket/run_opt.py>`:
  defines and launches the optimization with *optimas*.
- :download:`template_simulation_script.py <../../../examples/multitask_lpa_fbpic_waket/template_simulation_script.py>`:
  template used by *optimas* to generate the FBPIC and Wake-T simulation
  scripts.
- :download:`analysis_script.py <../../../examples/multitask_lpa_fbpic_waket/analysis_script.py>`:
  defines how the simulation data is analyzed to yield the value of the
  objective function.
- :download:`bunch_utils.py <../../../examples/multitask_lpa_fbpic_waket/bunch_utils.py>`:
  contains methods for generating the beam particle distributions given to the
  simulations.
- :download:`custom_fld_diags.py <../../../examples/multitask_lpa_fbpic_waket/custom_fld_diags.py>`:
  custom FBPIC field diagnostics that have been
  to generate the output with the same location and periodicity as
  Wake-T.
- :download:`custom_ptcl_diags.py <../../../examples/multitask_lpa_fbpic_waket/custom_ptcl_diags.py>`:
  custom FBPIC particle diagnostics that have been
  modified to generate the output with the same location and periodicity as
  Wake-T.

You can have a look at the main scripts below:

.. literalinclude:: ../../../examples/multitask_lpa_fbpic_waket/run_opt.py
   :language: python
   :caption: run_opt.py (:download:`download <../../../examples/multitask_lpa_fbpic_waket/run_opt.py>`)

.. literalinclude:: ../../../examples/multitask_lpa_fbpic_waket/template_simulation_script.py
   :language: python
   :caption: template_simulation_script.py (:download:`download <../../../examples/multitask_lpa_fbpic_waket/template_simulation_script.py>`)

.. literalinclude:: ../../../examples/multitask_lpa_fbpic_waket/analysis_script.py
   :language: python
   :caption: analysis_script.py (:download:`download <../../../examples/multitask_lpa_fbpic_waket/analysis_script.py>`)
