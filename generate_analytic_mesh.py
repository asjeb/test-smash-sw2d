import smash
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rasterio.crs import CRS

# Bristeau 2005
# import math
# R =  6378137.0
# def y2lat(y):
#     return np.degrees(2 * np.atan(np.exp (y / R)) - np.pi / 2.0)
# def lat2y(lat):
#     return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * R
# def x2lng(x):
#     return np.degrees(x / R)
# def lon2x(lon):
#     return math.radians(lon) * R


l = 10.
L = 10.
N = 1000
M = 1000
x = np.linspace(0, L, N)
y = np.linspace(0, l, M)
X, Y = np.meshgrid(x, y)
Z = 3 * np.ones(X.shape)
import rasterio
from rasterio.transform import Affine
xres = (x[-1] - x[0]) / (len(x))
yres = (y[-1] - y[0]) / (len(y))
print(xres, yres)
transform = rasterio.transform.from_bounds(0, 0, L, l, M, N)
# transform = Affine.scale(xres, yres)
print(transform)
with rasterio.open(
        "./ratafia.tif",
        mode="w",
        driver="GTiff",
        height=Z.shape[0],
        width=Z.shape[1],
        count=1,
        dtype=Z.dtype,
        crs=CRS.from_string("EPSG:2154"),
        transform=transform,
) as new_dataset:
        new_dataset.write(Z, 1)


mesh = smash.factory.generate_mesh(
    flwdir_path="./ratafia.tif",
    bbox=[0, L, 0, l],
)
print(mesh["xres"], mesh["yres"])
print(mesh["dx"], mesh["dy"])

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