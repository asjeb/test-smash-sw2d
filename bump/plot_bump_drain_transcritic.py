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


with h5py.File("transcritical_bump.hdf5", "r") as f:
    X = f["X"][()]
    Y = f["Y"][()]
    topography = f["topography"][()]
    hsw = f["hsw"][()]
    eta = f["eta"][()]
    qx = f["qx"][()]
    qy = f["qy"][()]
    times = f["times"][()]

# print(eta[0, :, 950])
fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.fill(X, topography[0, :], label="topography", color = "grey")
plt.plot(X, eta[0, :, 0], '-', label="t = {}".format(times[0]), color='black')
# plt.plot(X, eta[0, :, 20], '--', label="t = {}".format(times[20]), color='black')
# # plt.plot(X, eta[0, :, 100], '+', label="t = {}".format(times[100]), color='black')
# # plt.plot(X, eta[0, :, 310], '-', label="t = {}".format(times[310]), color='black')
# # plt.plot(X, eta[0, :, 400], '-.', label="t = {}".format(times[400]), color='black')
plt.plot(X, eta[0, :, 500], '--', label="t = {}".format(times[500]), color='black')
# plt.plot(X, eta[0, :, 1800], '--', label="t = {}".format(times[1800]), color='black')
plt.plot(X, eta[0, :, 2500], '+', label="t = {}".format(times[2500]), color='black')

# plt.plot(X, eta[0, :, 3000], '-', label="t = {}".format(times[3000]), color='black')
# plt.plot(X, eta[0, :, 4000], '--', label="t = {}".format(times[4000]), color='black')
# plt.plot(X, eta[0, :, 4500], '-.', label="t = {}".format(times[4500]), color='black')
# plt.plot(X, eta[0, :, 5000], '-.', label="t = {}".format(times[5000]), color='black')
# plt.plot(X, eta[0, :, 3000], '+', label="t = {}".format(times[3000]), color='black')


# plt.plot(X, eta[0, :, 1000], '+', label="computed water height", color='black')

plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.legend(loc='upper right')
plt.show()