import smash
import numpy as np
import matplotlib.pyplot as plt
from analytic_mesh import analytic_mesh


l = 4.
L = 20.

N = 60 # pixel
M = 12 # pixel

mesh = analytic_mesh.generate_analytic_mesh(L, l, N, M, "bump")

setup = {
    "start_time": "1951-05-17 00:00",
    "end_time": "1951-05-24 00:00",
    "dt": 8000,
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

he = np.empty(len(X))
qxe = np.empty(len(X))
qye = np.empty(len(X))
topoe = np.empty(len(X))
for i, x in enumerate(X):    
    coef = cardan(9.81, topo(x), 4.42, 2.0)
    he[i] = max(np.roots(coef).real)

# print(eta[0, :, 950])
fig, ax = plt.subplots(1, 1, figsize=(8,15))
ax.fill(X, topography[0, :], label="topography", color = "grey")
plt.plot(X, eta[0, :, 0], '-', label="t = 0", color='black')
plt.plot(X, eta[0, :, 20], '--', label="t = 20", color='black')
plt.plot(X, eta[0, :, 100], '+', label="t = 100", color='black')
plt.plot(X, eta[0, :, 310], '-', label="t = 310", color='black')
plt.plot(X, eta[0, :, 600], '-.', label="t = 600", color='black')
plt.plot(X, eta[0, :, 900], '--', label="t = 900", color='black')
plt.plot(X, eta[0, :, 1800], '+', label="t = 1000", color='black')


# plt.plot(X, eta[0, :, 1000], '+', label="computed water height", color='black')

plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.legend()
plt.show()


