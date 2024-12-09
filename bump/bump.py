import smash
import numpy as np
import matplotlib.pyplot as plt
from analytic_mesh import analytic_mesh
import h5py


L = 20.
l = 4.

N = 60 # pixel
M = 12 # pixel

mesh = analytic_mesh.generate_analytic_mesh(L, l, N, M, "bump")

setup = {
    "start_time": "1951-05-17 00:00",
    "end_time": "1951-05-24 00:00",
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

X = np.linspace(0, L, N)
Y = np.linspace(0, l, M)

x_b = 10.
def topo(x):
    if 8. < x < 12.:
        topo = 0.2 - 0.05 * (x - x_b) ** 2
    else:
        topo = 0.0
    return topo

topography = np.array([topo(x) for x in X])
topography = np.transpose((np.repeat(topography, model.mesh.nrow)).reshape((model.mesh.ncol, model.mesh.nrow)))
model.set_rr_parameters("topography", topography)
topography = model.get_rr_parameters("topography")


res = model.forward_run()

hsw = res.sw2d["hsw"]
eta = res.sw2d["eta"]
qx = res.sw2d["qx"]
qy = res.sw2d["qy"]


def cardan(gravity, topography, q_in, h_out):
    coef = np.empty((4))
    coef[0] = 1.0
    coef[1] = topography - q_in ** 2 / (2. * gravity * h_out ** 2) - h_out
    coef[2] = 0.0
    coef[3] = q_in ** 2 / (2. * gravity)
    return coef



topoe = np.empty(len(X)) 
he = np.empty(len(X))
for i, x in enumerate(X):    
    coef = cardan(9.81, topo(x), 0., 2.0)
    he[i] = max(np.roots(coef).real)


he_full_sw = np.empty(len(X))
for i, x in enumerate(X):    
    coef = cardan(9.81, topo(x), 4.42, 2.0)
    he_full_sw[i] = max(np.roots(coef).real)



def relative_error(exact_u, computed_u, ord=None):
    norm_du = np.linalg.norm(exact_u - computed_u, ord=ord)
    norm_ue = np.linalg.norm(exact_u, ord=ord)
    return norm_du / norm_ue


L2_error = relative_error(he, hsw[0, :, 10])
L2_error_full_sw = relative_error(he_full_sw, hsw[0, :, 10])
print(L2_error)
print(L2_error_full_sw)


with h5py.File("lake_at_rest_{}.hdf5".format(name), "w") as f:
    f.create_dataset("X", data=X)
    f.create_dataset("Y", data=Y)
    f.create_dataset("topography", data=topography)
    f.create_dataset("eta", data=eta)
    f.create_dataset("hsw", data=hsw)
    f.create_dataset("qx", data=qx)
    f.create_dataset("qy", data=qy)
    f.create_dataset("L2_error", data=L2_error)
    f.create_dataset("L2_error_full_sw", data=L2_error_full_sw)

