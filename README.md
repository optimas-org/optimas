[![PyPI](https://img.shields.io/pypi/v/optimas)](https://pypi.org/project/optimas/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/optimas.svg)](https://anaconda.org/conda-forge/optimas)
[![tests badge](https://github.com/optimas-org/optimas/actions/workflows/unix.yml/badge.svg)](https://github.com/optimas-org/optimas/actions)
[![Documentation Status](https://readthedocs.org/projects/optimas/badge/?version=latest)](https://optimas.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/287560975.svg)](https://zenodo.org/badge/latestdoi/287560975)
[![License](https://img.shields.io/pypi/l/optimas.svg)](license.txt)

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="https://user-images.githubusercontent.com/20479420/219680583-34ac9525-7715-4e2a-b4fe-74848e9f59b2.png" alt="optimas logo" width="350">
  </a>

  <h3 align="center">
    Optimization at scale, powered by
    <a href="https://libensemble.readthedocs.io/"><strong>libEnsemble</strong></a>
  </h3>

  <p align="center">
    <a href="https://optimas.readthedocs.io/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://optimas.readthedocs.io/en/latest/examples/index.html">View Examples</a>
    ·
    <a href="https://optimas-group.slack.com/">Support</a>
    ·
    <a href="https://optimas.readthedocs.io/en/latest/api/index.html">API Reference</a>
  </p>
</div>

Optimas is a Python library designed for highly scalable optimization, from laptops to massively-parallel supercomputers.


## Key Features

- **Scalability**: Leveraging the power of [libEnsemble](https://github.com/Libensemble/libensemble), Optimas is designed to scale seamlessly from your laptop to high-performance computing clusters.
- **User-Friendly**: Optimas simplifies the process of running large parallel parameter scans and optimizations. Specify the number of parallel evaluations and the computing resources to allocate to each of them and Optimas will handle the rest.
- **Advanced Optimization**: Optimas integrates algorithms from the [Ax](https://github.com/facebook/Ax) library, offering both single- and multi-objective Bayesian optimization. This includes advanced techniques such as multi-fidelity and multi-task algorithms.


## Installation
You can install Optimas from PyPI (recommended):
```sh
pip install optimas
```
from conda-forge:
```sh
conda install optimas --channel conda-forge
```
or directly from GitHub:
```sh
pip install git+https://github.com/optimas-org/optimas.git
```
Make sure `mpi4py` is available in your environment before installing optimas. Fore more details, check out the full [installation guide](https://optimas.readthedocs.io/en/latest/user_guide/installation_local.html). We have also prepared dedicated installation instructions for some HPC systems such as
[JUWELS (JSC)](https://optimas.readthedocs.io/en/latest/user_guide/installation_juwels.html),
[Maxwell (DESY)](https://optimas.readthedocs.io/en/latest/user_guide/installation_maxwell.html) and
[Perlmutter (NERSC)](https://optimas.readthedocs.io/en/latest/user_guide/installation_perlmutter.html).


## Documentation
For more information on how to use Optimas, check out the [documentation](https://optimas.readthedocs.io/). You'll find installation instructions, a user guide, [examples](https://optimas.readthedocs.io/en/latest/examples/index.html) and the API reference.


## Support
Need more help? Join our [Slack channel](https://optimas-group.slack.com/) or open a [new issue](https://github.com/optimas-org/optimas/issues/new/choose).


## Citing optimas
If your usage of Optimas leads to a scientific publication, please consider citing the original [paper](https://link.aps.org/doi/10.1103/PhysRevAccelBeams.26.084601):
```bibtex
@article{PhysRevAccelBeams.26.084601,
    title     = {Bayesian optimization of laser-plasma accelerators assisted by reduced physical models},
    author    = {Ferran Pousa, A. and Jalas, S. and Kirchen, M. and Martinez de la Ossa, A. and Th\'evenet, M. and Hudson, S. and Larson, J. and Huebl, A. and Vay, J.-L. and Lehe, R.},
    journal   = {Phys. Rev. Accel. Beams},
    volume    = {26},
    issue     = {8},
    pages     = {084601},
    numpages  = {9},
    year      = {2023},
    month     = {Aug},
    publisher = {American Physical Society},
    doi       = {10.1103/PhysRevAccelBeams.26.084601},
    url       = {https://link.aps.org/doi/10.1103/PhysRevAccelBeams.26.084601}
}
```
and libEnsemble:
```bibtex
@article{Hudson2022,
    title   = {{libEnsemble}: A Library to Coordinate the Concurrent
                Evaluation of Dynamic Ensembles of Calculations},
    author  = {Stephen Hudson and Jeffrey Larson and John-Luke Navarro and Stefan M. Wild},
    journal = {{IEEE} Transactions on Parallel and Distributed Systems},
    volume  = {33},
    number  = {4},
    pages   = {977--988},
    year    = {2022},
    doi     = {10.1109/tpds.2021.3082815}
}
```
