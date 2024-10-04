import smash
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from rasterio.crs import CRS
import os

# Bristeau 2005

l = 4.
L = 20.
N = 60
M = 12

def generate_analytical_mesh(length, width, nx, ny, filename):
    x = np.linspace(0, length, nx)
    y = np.linspace(0, width, ny)
    X, Y = np.meshgrid(x, y)
    Z = 3 * np.ones(X.shape)

    xres = (x[-1] - x[0]) / (len(x))
    yres = (y[-1] - y[0]) / (len(y))

    transform = rasterio.transform.from_bounds(0, 0, length, width, nx, ny)

    if not os.path.exists("mesh"):
        os.makedirs("mesh")

    with rasterio.open(
        "./mesh/{}.tif".format(filename),
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
        flwdir_path="./mesh/{}.tif".format(filename),
        bbox=[0, length, 0, width],
    )

    return mesh


mesh = generate_analytical_mesh(L, l, N, M, "bump")

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