import smash
import numpy as np
import matplotlib.pyplot as plt
from analytic_mesh import analytic_mesh

L = 10.
l = 0.5

N = 200 # pixel
M = 10 # pixel

mesh = analytic_mesh.generate_analytic_mesh(L, l, N, M, "dambreak")

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

res = model.forward_run()

hsw = res.sw2d["hsw"]
eta = res.sw2d["eta"]
qx = res.sw2d["qx"]
qy = res.sw2d["qy"]

X = np.linspace(0, L, N)

x0 = 5.
gravity = 9.81

hl = 0.005
hr = 0.001

coeff = np.zeros(7)

coeff[0] = 1.0
coeff[1] = 0.0
coeff[2] = -9. * gravity * hr
coeff[3] = 16. * gravity * hr * np.sqrt(gravity * hl)
coeff[4] = gravity ** 2 * hr * (-8. * hl - hr)
coeff[5] = 0.0
coeff[6] = (gravity * hr) ** 3

roots = np.roots(coeff)
cm = max(roots.real)

times = res.sw2d_times 
times = times - np.min(times)

Xa = np.zeros(times.shape)
Xb = np.zeros(times.shape)
Xc = np.zeros(times.shape)
for i, t in enumerate(times):
    Xa[i] = x0 - t * np.sqrt(gravity * hl)
    Xb[i] = x0 + t * (2 * np.sqrt(gravity * hl) - 3 * cm)
    Xc[i] = x0 + t * (2 * cm ** 2 * (np.sqrt(gravity * hl) - cm)) / (cm ** 2 - gravity * hr)

def h_ex(i, t):
    if X[i] <= Xa[t]:
        return hl
    
    if Xa[t] <= X[i] <= Xb[t]:
        return 4. / (9. * gravity) * (np.sqrt(gravity * hl) - (X[i] - x0) / (2. * times[t])) ** 2.
    
    if Xb[t] <= X[i] <= Xc[t]:
        return cm ** 2. / gravity
    
    if Xc[t] <= X[i]:
        return hr

def u_ex(i, t):
    if X[i] <= Xa[t]:
        return 0.
    
    if Xa[t] <= X[i] <= Xb[t]:
        return (2. / 3.) * (X[i] - x0) / times[t] + np.sqrt(gravity * hl)

    if Xb[t] <= X[i] <= Xc[t]:
        return 2. * (np.sqrt(gravity * hl) - cm)
    
    if Xc[t] <= X[i]:
        return 0.

def qx_ex(i, t):
    return h_ex(i, t) * u_ex(i, t)

he = np.zeros((len(X), len(times)))
qxe = np.zeros((len(X), len(times)))
for i in range(len(X)):
    for t in range(len(times)):
        he[i, t] = h_ex(i, t)
        qxe[i, t] = h_ex(i, t) * u_ex(i, t)

t1=5
t2=50
t3=100
plt.plot(X, eta[0, :, t1], '+', label="computed free surface - time {:.1f} s".format(times[t1]), color='black')
plt.plot(X, eta[0, :, t2], '+', label="computed free surface - time {:.1f} s".format(times[t2]), color='grey')
plt.plot(X, eta[0, :, t3], '+', label="computed free surface - time {:.1f} s".format(times[t3]), color='tan')

plt.plot(X, he[:, t1],'-', label="exact free surface - time {:.1f} s".format(times[t1]), color='black')
plt.plot(X, he[:, t2],'-', label="exact free surface - time {:.1f} s".format(times[t2]), color='grey')
plt.plot(X, he[:, t3],'-', label="exact free surface - time {:.1f} s".format(times[t3]), color='tan')
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.legend()
plt.show()


plt.plot(X, qx[0, :, t1], '+', label="computed flowrate - time {:.1f} s".format(times[t1]), color='black')
plt.plot(X, qx[0, :, t2], '+', label="computed flowrate - time {:.1f} s".format(times[t2]), color='grey')
plt.plot(X, qx[0, :, t3], '+', label="computed flowrate - time {:.1f} s".format(times[t3]), color='tan')

plt.plot(X, qxe[:, t1],'-', label="exact flowrate - time {:.1f} s".format(times[t1]), color='black')
plt.plot(X, qxe[:, t2],'-', label="exact flowrate - time {:.1f} s".format(times[t2]), color='grey')
plt.plot(X, qxe[:, t3],'-', label="exact flowrate - time {:.1f} s".format(times[t3]), color='tan')

plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.legend()
plt.show()