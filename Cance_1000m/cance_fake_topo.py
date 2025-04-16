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
from smash.factory.mesh._tools import _get_transform
# from rasterio.crs import CRS


setup = {
    "start_time": "2014-09-15 00:00",
    "end_time": "2014-11-14 00:00",
    "dt": 3_600,
    "hydrological_module": "gr4",
    "routing_module": "sw2d",
    "read_qobs": True,
    "qobs_directory": "./Cance-dataset/qobs",
    "read_prcp": True,
    "prcp_conversion_factor": 0.1,
    "prcp_directory": "./Cance-dataset/prcp",
    "read_pet": True,
    "daily_interannual_pet": True,
    "pet_directory": "./Cance-dataset/pet",
    "read_descriptor": False,
    "read_bathymetry": True,
    "bathymetry_format":"tif",
    "bathymetry_file":"./fake_topography.tif"
}

gauge_attributes = pd.read_csv("./Cance-dataset/gauge_attributes.csv")
mesh = smash.factory.generate_mesh(
    flwdir_path="./Cance-dataset/France_flwdir.tif",    
    x=list(gauge_attributes["x"]),
    y=list(gauge_attributes["y"]),
    area=list(gauge_attributes["area"] * 1e6), # Convert km² to m²
    code=list(gauge_attributes["code"]),
)

model = smash.Model(setup, mesh)

# print(model.physio_data.bathymetry)


res = model.forward_run(
    return_options={
    "internal_fluxes":True,
    "q_domain":True,
    },
)

if model.setup.routing_module == "sw2d":

    hsw = res.sw2d["hsw"]
    eta = res.sw2d["eta"]
    qx = res.sw2d["qx"]
    qy = res.sw2d["qy"]
    qt = res.internal_fluxes["qt"]
    times = res.sw2d_times

    with h5py.File("hsw.hdf5", "w") as f:
        f.create_dataset("hsw", data=hsw)
    with h5py.File("eta.hdf5", "w") as f:
        f.create_dataset("eta", data=eta)
    with h5py.File("qx.hdf5", "w") as f:
        f.create_dataset("qx", data=qx)
    with h5py.File("qy.hdf5", "w") as f:
        f.create_dataset("qy", data=qy)


# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# plt.imshow(model.physio_data.bathymetry)
# plt.colorbar(label="z")
# plt.title("Topography")

# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# plt.imshow(model.physio_data.bathymetry)
# plt.colorbar(label="z")
# plt.title("Topography")

# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# plt.imshow(hsw[:,:,0])
# plt.colorbar(label="z")
# plt.title("Water height")

# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# plt.imshow(eta[:,:,0])
# plt.colorbar(label="z")
# plt.title("Free surface")
# plt.show()

for i in range(20):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.imshow(hsw[:, :, i])
    plt.colorbar(label="z")
plt.show()

# outlet_indices = model.mesh.outlet_indices
# print(outlet_indices)
# base = np.zeros(shape=(mesh["nrow"], mesh["ncol"]))
# base = np.where(mesh["active_cell"] == 0, np.nan, base)
# for pos in mesh["gauge_pos"]:
#     base[outlet_indices[0], outlet_indices[1]] = 1

# plt.imshow(base, cmap="Set1_r");
# plt.show()

# boundaries = model.mesh.boundaries
# plt.imshow(boundaries)
# plt.colorbar()
# plt.show()
