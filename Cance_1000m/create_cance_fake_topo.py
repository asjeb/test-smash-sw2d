import smash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import pickle
import matplotlib.colors as mcolors
import os
import pyflwdir
import rasterio
from rasterio.enums import Resampling
# from rasterio.crs import CRS


setup = {
    "start_time": "2014-09-15 00:00",
    "end_time": "2014-09-15 01:00",
    "dt": 3_600,
    "hydrological_module": "gr4",
    "routing_module": "lr",
    "read_qobs": True,
    "qobs_directory": "./Cance-dataset/qobs",
    "read_prcp": True,
    "prcp_conversion_factor": 0.1,
    "prcp_directory": "./Cance-dataset/prcp",
    "read_pet": True,
    "daily_interannual_pet": True,
    "pet_directory": "./Cance-dataset/pet",
    "read_descriptor": False,
}


gauge_attributes = pd.read_csv("./Cance-dataset/gauge_attributes.csv")
mesh = smash.factory.generate_mesh(
    flwdir_path="./Cance-dataset/France_flwdir.tif",    
    x=list(gauge_attributes["x"]),
    y=list(gauge_attributes["y"]),
    area=list(gauge_attributes["area"] * 1e6), # Convert km² to m²
    code=list(gauge_attributes["code"]),
)

transform = rasterio.transform.Affine(
    mesh['xres'], 0, mesh["xmin"], 
    0, -mesh["yres"], mesh["ymax"], 
)

print(transform)

    #  transform = rasterio.transform.Affine(
    #             xres, 0, xmin + slice_win[1].start * xres, 0, -yres, ymax - slice_win[0].start * yres
    #         )


flwdst = mesh["flwdst"]
flwdst_copy = flwdst.copy()
flwdst_copy[flwdst_copy != -99] *= 0.0001

filled_topography = flwdst_copy


with rasterio.open("./Cance-dataset/France_flwdir.tif", "r") as dataset:
    transform = transform
    crs = dataset.crs    
    with rasterio.open(
        'fake_topography.tif',
        'w',
        driver='GTiff',
        height=filled_topography.shape[0],
        width=filled_topography.shape[1],
        count=1,
        dtype=filled_topography.dtype,
        crs=crs, #CRS.from_epsg(2154),
        transform=transform,
        nodata=-99,
    ) as dst:
        dst.write(filled_topography, 1)




with rasterio.open("fake_topography.tif", "r") as dataset:
    topoto = dataset.read(1)

smash.io.save_setup(setup, "cance_setup.yaml")
smash.io.save_mesh(mesh, "cance_mesh.hdf5")

topoto = np.where(
    topoto == -99,
    np.nan,
    topoto
)
# plt.figure()
# plt.imshow(topoto)
# plt.colorbar()

# plt.figure()
# plt.imshow(mesh["active_cell"]);
# plt.show()


# mask = mesh["active_cell"]
# n, m = mask.shape


# coloring = -99 * np.ones((n, m))
# for j in range(n):
#     for i in range(m):
#         if coloring[i, j] != 2 or coloring[i, j] != 1:

#             if mask[i, j] == 1:
#                 coloring[i, j] = 2 # already pass

#                 if j < n - 1:
#                     if coloring[i, j + 1] != 2 or coloring[i, j + 1] != 1:
#                         if mask[i, j + 1] == 0:
#                             coloring[i, j] = 1 
#                 if j > 0:
#                     if coloring[i, j - 1] != 2 or coloring[i, j - 1] != 1:
#                         if mask[i, j - 1] == 0:
#                             coloring[i, j] = 1
#                 if i < m - 1:
#                     if coloring[i + 1, j] != 2 or coloring[i + 1, j] != 1:
#                         if mask[i + 1, j] == 0:
#                             coloring[i, j] = 1
#                 if i > 0:
#                     if coloring[i - 1, j] != 2 or coloring[i - 1, j] != 1:
#                         if mask[i - 1, j] == 0:
#                             coloring[i, j] = 1

# coloring[coloring==2] = -99

plt.figure()
plt.imshow(mesh["boundaries"]);
plt.show()
# print(mesh.boundaries)