Original release of model in conjunction with publication:

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.48474.svg)](http://dx.doi.org/10.5281/zenodo.48474)


This model and its results are discussed in an article avaiable through the Journal of Geophysical Research (http://onlinelibrary.wiley.com/doi/10.1002/2016JD025159/full).

Please cite:

V. C. Chmielewski and E. C. Bruning (2016), Lightning Mapping Array flash detection performance with variable receiver thresholds, J. Geophys. Res. Atmos., 121, 8600-8614, doi:10.1002/2016JD025159

If any results from this model are presented.

# LMAsimulation
This code is designed to simulate the way any LMA would solve a grid of point emitters, accounting for each station's noise levels and adding normally distributed timing errors at each station.

### LMAsimulation_full.ipynb

This is the primary notebook for plotting the errors in the solutions over a regular, gridded domain in a Monte Carlo model. The used stations and thresholds can be taken from a time range of station log files (with any set of stations exempted) or from a list of stations in the network.csv file.

The grid of points is set in 'initial_points' in x,y,z in the map projection plane. The solutions are currently all found through scipy's lstsq function. The function can also be run for several iterations at a singular location.

The flash detection efficiency is calculated from the WTLMA climatology results in fde.csv. The flash area error estimate is calculated from the typical_flashes.csv file which contains the median flash size and number of points of flashes with at least x+9 number of points (per line).

Note: The log file reading function is set up to handle errors in files through Linux subprocesses, so any errors may cause it to crash on other platforms. The errored logs are copied to an '*_original' file and the bad lines are removed from the active file.

### CurvatureMatrix.ipynb

This runs the curvature matrix calculations for rmse of solution points in a gridded domain. The used stations must be specified in the network.csv file.

### simulation_function.py

This contains the calculations for all models.

### read_logs.py

This is intended for parsing, quick QC, and integrating v10 log files for LMA stations.

### coordinateSystems.py

This contains the driving functions for the coordinate system transformations

### network_full.csv

This csv contains the approximated station locations and thresholds used for analysis. For more precise station locations, please contact the inidividual network operators or the authors of this program.