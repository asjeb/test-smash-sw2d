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

print(model.physio_data.bathymetry)
