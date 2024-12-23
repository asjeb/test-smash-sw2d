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
flwdst_copy[flwdst_copy != -99] *= 0.00001

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
plt.figure()
plt.imshow(topoto)
plt.colorbar()
plt.show()


'''
manning = 0.033
model = smash.Model(setup, mesh)
if model.setup.routing_module == "sw2d":
    model.set_rr_parameters("topography", filled_topography)
    topo = model.get_rr_parameters("topography")

topo = np.where(
    topo == -99,
    np.nan,
    topo
)
# print(topo == filled_topography)
# plt.figure()
# plt.imshow(topo)
# plt.colorbar()


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
    topography = res.sw2d["topography"]
    manning = res.sw2d["manning"]
    times = res.sw2d_times

    with h5py.File("hsw.hdf5", "w") as f:
        f.create_dataset("hsw", data=hsw)
    with h5py.File("eta.hdf5", "w") as f:
        f.create_dataset("eta", data=eta)
    with h5py.File("qx.hdf5", "w") as f:
        f.create_dataset("qx", data=qx)
    with h5py.File("qy.hdf5", "w") as f:
        f.create_dataset("qy", data=qy)

print(eta.shape)
plt.figure()
plt.imshow(eta[:,:,0])
plt.colorbar()

plt.figure()
plt.imshow(topography[:,:,0])
plt.imshow(manning[:,:,0])

plt.colorbar()
plt.show()
'''