import smash
import numpy as np
import matplotlib.pyplot as plt
from analytic_mesh import analytic_mesh
import h5py

L = 1000.
l = 10.

N = 100 # pixels
M = 1 # pixels

mesh = analytic_mesh.generate_analytic_mesh(L, l, N, M, "macdo")

setup = {
    "start_time": "1951-05-17 00:00",
    "end_time": "1951-05-30 00:00",
    "dt": 7200,
    "routing_module": "sw2d",
    "read_prcp": False,
    "prcp_directory": "./prcp",
    "prcp_access": "%Y/%m/%d",
    "read_pet": False,
    "pet_directory": "./pet",
    "prcp_access": "%Y/%m/%d",
}

model = smash.Model(setup, mesh)

def hsw_ex(x):
    gravity = 9.81
    return (4. / gravity) ** (1. / 3) * (1 + 0.5 * np.exp(-16 * (x / 1000. - 0.5) ** 2))

def Dhsw_ex(x):
    gravity = 9.81
    return -(4. / gravity) ** (1. / 3) ** 2. / 125 * (x / 1000. - 0.5) * np.exp(-16 * (x / 1000. - 0.5) ** 2)

def topo1(X, n):
    N = len(X)

    he = hsw_ex(X)
    Dhe = Dhsw_ex(X)
    q = 2.

    gravity = 9.81

    z = np.empty(N)
    z[0] = 0.

    dx = X[1:] - X[:-1]
    print(dx)
    for i in range(N-1):
        z[i+1] = z[i] + dx[i] * ((q ** 2 / gravity / he[i] ** 3 - 1) * Dhe[i] - (n ** 2 * q ** 2) / he[i] ** (10. / 3))

    z = (z + abs(np.min(z)))
    return z

def topo2(X, n):
    N = len(X)

    he = hsw_ex(X)
    Dhe = Dhsw_ex(X)
    q = 2.

    gravity = 9.81

    z = np.empty(N)
    z[0] = 0.

    dx = X[1:] - X[:-1]
    print(dx)
    for i in range(N-1):
        z[i+1] = z[i] + dx[i] * ((- 1) * Dhe[i] - (n ** 2 * q ** 2) / he[i] ** (10. / 3))

    z = (z + abs(np.min(z)))
    return z

X = np.linspace(0, L, N)
Y = np.linspace(0, l, M)

manning = 0.033
model.set_rr_parameters("manning", manning)
topography = topo2(X, manning)
print(topography)
topography = np.transpose((np.repeat(topography, model.mesh.nrow)).reshape((model.mesh.ncol, model.mesh.nrow)))
model.set_rr_parameters("topography", topography)
topography = model.get_rr_parameters("topography")

he = hsw_ex(X)

res = model.forward_run()

hsw = res.sw2d["hsw"]
eta = res.sw2d["eta"]
qx = res.sw2d["qx"]
qy = res.sw2d["qy"]
times = res.sw2d_times
times = times - times[0]

with h5py.File("macdonald.hdf5", "w") as f:
    f.create_dataset("X", data=X)
    f.create_dataset("Y", data=Y)
    f.create_dataset("times", data=times)
    f.create_dataset("topography", data=topography)
    f.create_dataset("eta", data=eta)
    f.create_dataset("hsw", data=hsw)
    f.create_dataset("qx", data=qx)
    f.create_dataset("qy", data=qy)
    f.create_dataset("he", data=he)
