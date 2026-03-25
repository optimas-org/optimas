## Structure exploiting Optimization using APOSMM and IBCDFO

This optimization uses IBCDFO POUNDERS to perform structure-exploiting
optimization of a laser-plasma accelerator beam. Instead of treating the
objective as a single black box, IBCDFO builds separate surrogate models for
each of the 21 simulation observables (charge, 10 bin gammas, and 10 bin
particle counts), then uses the known mathematical relationship defined in hfun
to combine these models into a single objective model. This approach is more
sample-efficient than standard derivative-free optimization because it leverages
the structure of how the observables combine, requiring fewer expensive
simulations to find optimal beam parameters.

## Requirements

Requires IBCDFO and JAX for optimization and wake-t for the simulation:

    pip install wake-t
    pip install jax

Install IBCDFO and add minq5 to PYTHONPATH:

    git clone git@github.com:POptUS/IBCDFO.git
    cd IBCDFO/ibcdfo_pypkg
    git checkout main
    pip install -e .
    cd ..
    git submodule update --init --recursive
    cd minq/py/minq5
    export PYTHONPATH="$PYTHONPATH:$(pwd)"
