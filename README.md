# Interval Signal Temporal Logic for Robust Optimal Control

This repository accompanies the paper submitted to CDC 2024 "Interval Signal Temporal Logic for Robust Optimal Control," providing code to generate the figures in our paper.

This code builds on `stlpy`. See [stlpy](https://stlpy.readthedocs.io/en/latest/)'s documentation.

## Miniature Blimp Case Study
Requires Python >=3.9.
Install the following packages
- numpy
- matplotlib
- scipy
- [npinterval](https://github.com/gtfactslab/npinterval)

Follow the instructions to install [gurobi](https://www.gurobi.com/downloads/free-academic-license/) for Python. It is free for academia. Next,
- Clone [the stlpy repo](https://github.com/vincekurtz/stlpy)  
- Add `gurobi_optimal_control` to the folder `stlpy/solvers/gurobi/` 
- Replace `stlpy/STL/predicate.py` with the one in this repo  
- Replace `stlpy/STL/formula.py` with the one in this repo  
- Add `from .gurobi.gurobi_optimal_control import GurobiIntervalOptimalControl` in the `GUROBI_ENABLED` section of `stlpy/solvers/__init__.py`  
- Run `pip install .` from the home directory of `stlpy`.
- Run `python final-paper-code.py`.

## Latex Installation (Optional)
- [MikTex](https://miktex.org/) or another LaTeX interpreter, for LaTeX to appear in PyPlot plots.
