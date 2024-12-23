import smash
import numpy as np
import matplotlib.pyplot as plt
import h5py

with h5py.File("eta.hdf5", "r") as f:
    eta = f["eta"][()]

with h5py.File("X.hdf5", "r") as f:
    X = f["X"][()]

with h5py.File("topography.hdf5", "r") as f:
    topography = f["topography"][()]

with h5py.File("times.hdf5", "r") as f:
    times = f["times"][()]

with h5py.File("hsw.hdf5", "r") as f:
    hsw = f["hsw"][()]

plt.figure()
# for i in range(4990, 5000):
#     plt.plot(X, eta[49, :, i], '+', label="time = {} s".format(times[i]))
plt.plot(X, eta[49, :, 0], '+', label="time = {} s".format(times[0]))
plt.plot(X, eta[49, :, 40], '+', label="time = {} s".format(times[40]))
plt.plot(X, eta[49, :, 50], '+', label="time = {} s".format(times[50]))
plt.plot(X, eta[49, :, 60], '+', label="time = {} s".format(times[60]))
plt.plot(X, eta[49, :, 100], '+', label="time = {} s".format(times[100]))
plt.plot(X, eta[49, :, 200], '+', label="time = {} s".format(times[200]))
plt.plot(X, eta[49, :, 300], '+', label="time = {} s".format(times[300]))
# plt.plot(X, eta[49, :, 400], '+', label="time = {} s".format(times[400]))
# plt.plot(X, eta[49, :, 600], '+', label="time = {} s".format(times[400]))
# plt.plot(X, eta[49, :, 800], '+', label="time = {} s".format(times[400]))

plt.plot(X, topography[:, 49], '--', label="topography", color='black')

plt.xlabel("x (m)")
plt.ylabel("z (m)")

plt.legend()
plt.show()
