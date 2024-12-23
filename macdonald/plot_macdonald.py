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

def relative_error(exact_u, computed_u, ord=None):
    norm_du = np.linalg.norm(exact_u - computed_u, ord=ord)
    norm_ue = np.linalg.norm(exact_u, ord=ord)
    return norm_du / norm_ue


with h5py.File("macdonald.hdf5", "r") as f:
    X = f["X"][()]
    Y = f["Y"][()]
    times = f["times"][()]
    topography = f["topography"][()]
    topography_fortran = f["topography_fortran"][()]
    hsw = f["hsw"][()]
    eta = f["eta"][()]
    qx = f["qx"][()]
    qy = f["qy"][()]
    he = f["he"][()]


L2_error = relative_error(he+topography[0, :], eta[0, :, 800])
print(L2_error)
# 0.0035476079683886184 pour Dx = 1
# 0.0036814671588231767 pour Dx = 10    

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
plt.fill_between(
        x=X, 
        y1=topography[0, :], 
        color= "grey",
        label="topography")

plt.plot(X, eta[0, :, 800], '+', label=f"computed free surface \n relative L²-error = {L2_error:.1e}", color='black')

plt.plot(X, he+topography[0, :], label="exact free surface", color='black')
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title("time = {} s".format(int(times[800])))
plt.legend()

print(topography.shape)
print(topography_fortran.shape)

plt.figure()
plt.plot(X, topography[0, :], label="topo", color='black')
plt.plot(X, topography_fortran[0, :, 0], label="topo fortran", color='black')
plt.legend()

plt.figure()
plt.imshow(topography[:, :])
plt.colorbar()


plt.figure()
plt.imshow(topography_fortran[:, :, 0])
plt.colorbar()
plt.show()

