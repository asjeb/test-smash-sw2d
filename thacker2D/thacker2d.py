import smash
import numpy as np
import matplotlib.pyplot as plt
import h5py
from analytic_mesh import analytic_mesh

L = 4.
l = 4.

N = 100 # pixels
M = 100 # pixels

HSTAR = 0.1
A = 1.
B = 0.5

mesh = analytic_mesh.generate_analytic_mesh(L, l, N, M, "thacker")

setup = {
    "start_time": "1951-05-17 00:00",
    "end_time": "1951-05-18 00:00",
    "dt": 3600,
    "routing_module": "sw2d",
    "read_prcp": False,
    "prcp_directory": "./prcp",
    "prcp_access": "%Y/%m/%d",
    "read_pet": False,
    "pet_directory": "./pet",
    "prcp_access": "%Y/%m/%d",
}

model = smash.Model(setup, mesh)

def topo(x, y, hstar, a):
    r2 = (x - 0.5 * L) ** 2 + (y - 0.5 * L) ** 2
    return - hstar * (1 - (r2 / a ** 2))


def frequency(hstar, a):
    g = 9.81
    return np.sqrt(2 * g * hstar) / a


def h_ex(x, y, t, hstar, a, b):
    omega = frequency(hstar, a)
    return  2 * b * hstar / (a ** 2) * \
        ((x - L / 2) * np.cos(omega * t)\
        + (y - L / 2) * np.sin(omega* t)\
        - b / 2.) \
        - topo(x, y, hstar, a)
    

def u_ex(x, y, t, hstar, a, b):
    omega = frequency(hstar, a)
    return - b * omega * np.sin(omega * t)

def v_ex(x, y, t, hstar, a, b):
    omega = frequency(hstar, a)
    return b * omega * np.cos(omega * t)

X = np.linspace(0, L, N)
Y = np.linspace(0, L, M)

OMEGA = frequency(HSTAR, A)

p = 2 * 3.14 / OMEGA

T = np.linspace(0, 20, 10)

#T = [0., p / 6., 2 * p / 6, p / 2.]

# print(T)

he = np.zeros((len(X), len(Y), len(T)))
topography = np.zeros((len(X), len(Y)))
for i, x in enumerate(X):
    for j, y in enumerate(Y):
        for k, t in enumerate(T):
            he[i, j, k] = h_ex(x, y, t, HSTAR, A, B)
        topography[i, j] = topo(x, y, HSTAR, A)

topography = np.transpose(topography)
model.set_rr_parameters("topography", topography)
topography = model.get_rr_parameters("topography")
res = model.forward_run()

hsw = res.sw2d["hsw"]
eta = res.sw2d["eta"]
qx = res.sw2d["qx"]
qy = res.sw2d["qy"]
times = res.sw2d_times
print(times[0:100])

times = times - times[0]
print(times[0:100])

# with np.printoptions(threshold=np.inf):
#     print(times)


with h5py.File("hsw.hdf5", "w") as f:
    f.create_dataset("hsw", data=hsw)
with h5py.File("eta.hdf5", "w") as f:
    f.create_dataset("eta", data=eta)
with h5py.File("qx.hdf5", "w") as f:
    f.create_dataset("qx", data=qx)
with h5py.File("qy.hdf5", "w") as f:
    f.create_dataset("qy", data=qy)
with h5py.File("times.hdf5", "w") as f:
    f.create_dataset("times", data=times)
with h5py.File("topography.hdf5", "w") as f:
    f.create_dataset("topography", data=topography)
with h5py.File("X.hdf5", "w") as f:
    f.create_dataset("X", data=X)

# fig, ax = plt.subplots(1, 1, figsize=(8, 15))
# plt.fill_between(
#         x=X, 
#         y1=topography[:, 49], 
#         color= "grey",
#         label="topography")

free_surf = np.zeros((len(X), len(Y), len(T)))

for i, x in enumerate(X):
    for j, y in enumerate(Y):
        for k, t in enumerate(T):
            free_surf[i, j, k] = max(he[i, j, k] + topography[i, j], topography[i, j])

qex = np.zeros((len(X), len(Y), len(T)))
qey = np.zeros((len(X), len(Y), len(T)))

for i, x in enumerate(X):
    for j, y in enumerate(Y):
        for k, t in enumerate(T):
            qex[i, j, k] = max(0., u_ex(x, y, t, HSTAR, A, B) * he[i, j, k])
            qey[i, j, k] = max(0., v_ex(x, y, t, HSTAR, A, B) * he[i, j, k])


with h5py.File("exact_free_surf.hdf5", "w") as f:
    f.create_dataset("free_surf", data=free_surf)
with h5py.File("qex.hdf5", "w") as f:
    f.create_dataset("qex", data=qex)
with h5py.File("qey.hdf5", "w") as f:
    f.create_dataset("qey", data=qey)

