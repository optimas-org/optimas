[![PyPI](https://img.shields.io/pypi/v/optimas)](https://pypi.org/project/optimas/)
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

Optimas is a Python library for scalable optimization on massively-parallel supercomputers. See the [documentation](https://optimas.readthedocs.io/) for installation instructions, tutorials, and more information.

## Installation
From PyPI
```sh
pip install optimas
```
From GitHub
```sh
pip install git+https://github.com/optimas-org/optimas.git
```
Make sure `mpi4py` is available in your environment prior to installing optimas (see [here](https://optimas.readthedocs.io/en/latest/user_guide/installation_local.html) for more details).

Optimas is regularly used and tested in large distributed HPC systems.
We have prepared installation instructions for
[JUWELS (JSC)](https://optimas.readthedocs.io/en/latest/user_guide/installation_juwels.html),
[Maxwell (DESY)](https://optimas.readthedocs.io/en/latest/user_guide/installation_maxwell.html) and
[Perlmutter (NERSC)](https://optimas.readthedocs.io/en/latest/user_guide/installation_perlmutter.html).

## Citing optimas
If your usage of `optimas` leads to a scientific publication, please consider citing the original [paper](https://link.aps.org/doi/10.1103/PhysRevAccelBeams.26.084601):
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
