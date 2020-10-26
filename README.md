# Optimization of fbpic simulation with libEnsemble

The scripts in this repository allow to optimize FBPIC simulation.

## Contents of the repository

`optimization_folder/sim_specific`: Contains the scripts necessary to run the simulation with varying parameters. In particular, it has to contain the following 3 files:

- `template_fbpic_script.py`: an FBPIC script, templated with `jinja2` syntax.
- `varying_parameters.py`: list of varying parameters, along with their bounds. These variables should match the templated variables in `template_fbpic_script.py`.
- `analysis_script.py`: analyzes the result of the simulation and extract the objective function `f`.

This folder can be swapped out for other examples in `optimization_folder/example_sim_specific_folders`

`optimization_folder/run_libensemble.py`: this is the main file that controls the optimization.

## Installing and running

### On a local computer

Install FBPIC according to:
https://fbpic.github.io/install/install_local.html

Then install other dependencies:
```
pip install git+https://github.com/Libensemble/libensemble.git@feature/multi_fidelity
pip install -r requirements.txt
```

Then
```
git clone https://github.com/RemiLehe/fbpic_libE.git
cd fbpic_libE/optimization_folder
python run_libensemble.py --comms local --nworkers 3
```

### On Summit

Install according to:
https://fbpic.github.io/install/install_summit.html

Then install other dependencies:
```
pip install git+https://github.com/Libensemble/libensemble.git@feature/multi_fidelity
pip install -r requirements.txt
```

`cd` into your `$MEMBERWORK` folder, and create a dedicated directory. Then run:
```
git clone https://github.com/RemiLehe/fbpic_libE.git
cd fbpic_libE/optimization_folder
bsub submission_script
```
