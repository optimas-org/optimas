# Optimization of fbpic simulation with libEnsemble

The scripts in this repository allow to optimize FBPIC simulation.

## Installing

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
```

## Usage

`cd` into the folder `optimization_folder`, and run the script
`create_new_optimization.py`. (In order to see the usage of this script,
type `./create_new_optimization.py -h`.) Then follow the printed instructions.

Note that the script will create a new folder, with a number of files,
which you can modify before submitting/launching the optimization job:

- `template_fbpic_script.py`: an FBPIC script, templated with `jinja2` syntax.
- `varying_parameters.py`: list of varying parameters, along with their bounds. These variables should match the templated variables in `template_fbpic_script.py`.
- `mf_parameters.py` (optional): defines fidelity parameters for multi-fidelity optimization.
- `analysis_script.py`: analyzes the result of the simulation and extract the objective function `f`.
