import smash
import numpy as np
import matplotlib.pyplot as plt
from analytic_mesh import analytic_mesh
import h5py
size = 15
medium = 18
plt.rc('font', size=size)          # controls default text sizes
plt.rc('axes', titlesize=size)     # fontsize of the axes title
plt.rc('axes', labelsize=medium)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
plt.rc('legend', fontsize=size)    # legend fontsize
plt.rc('figure', titlesize=size)  # fontsize of the figure title

with h5py.File("lake_at_rest_emerged.hdf5", "r") as f:
    X = f["X"][()]
    Y = f["Y"][()]
    topography = f["topography"][()]
    hsw = f["hsw"][()]
    eta = f["eta"][()]
    qx = f["qx"][()]
    qy = f["qy"][()]

fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.fill(X, topography[0, :], label="topography", color = "grey")
plt.plot(X, eta[0, :, 10], '+', label="surface elevation", color='black')
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.legend()
plt.show()
