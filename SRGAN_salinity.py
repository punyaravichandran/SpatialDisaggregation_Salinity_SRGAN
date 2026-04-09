# Enable GPU
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU detected:", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU detected")

# Mixed Precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Imports
import earthaccess
import os
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import rioxarray
import scipy.ndimage
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Authenticate NASA Earthdata
earthaccess.login(persist=True)

# Search datasets
tspan = ("2022-06-05", "2022-06-10")

sss_results = earthaccess.search_data(short_name="OISSS_L4_multimission_7day_v2", temporal=tspan)
wind_results = earthaccess.search_data(short_name="SMAP_RSS_L3_SSS_SMI_8DAY-RUNNINGMEAN_V6", temporal=tspan)
sst_results = earthaccess.search_data(short_name="MUR-JPL-L4-GLOB-v4.1", temporal=tspan)
ssh_results = earthaccess.search_data(short_name="SEA_SURFACE_HEIGHT_ALT_GRIDS_L4_2SATS_5DAY_6THDEG_V_JPL2205", temporal=tspan)

# Open datasets
files_sss = earthaccess.open(sss_results)
files_sst = earthaccess.open(sst_results)
files_ssh = earthaccess.open(ssh_results)
files_wind = earthaccess.open(wind_results)

# Preprocess function
def time_from_attr(ds):
    dt_str = ds.attrs["time_coverage_start"]
    ds["date"] = ((), pd.to_datetime(dt_str))
    ds = ds.set_coords("date")

    lat_min, lat_max = 2, 26
    lon_min, lon_max = 64, 91

    if 'lat' in ds.dims and 'lon' in ds.dims:
        return ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    elif 'Latitude' in ds.dims:
        return ds.sel(Latitude=slice(lat_min, lat_max), Longitude=slice(lon_min, lon_max))
    elif 'latitude' in ds.dims:
        return ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    else:
        return ds

# Load datasets
dataset1 = xr.open_mfdataset(files_sss, preprocess=time_from_attr, combine="nested", concat_dim="date")
dataset2 = xr.open_mfdataset(files_sst, preprocess=time_from_attr, combine="nested", concat_dim="date")
dataset4 = xr.open_mfdataset(files_wind, preprocess=time_from_attr, combine="nested", concat_dim="date")

dataset3 = None
if files_ssh:
    dataset3 = xr.open_mfdataset(files_ssh, preprocess=time_from_attr, combine="nested", concat_dim="date")

# Copernicus data
import copernicusmarine

forecast_sss = copernicusmarine.open_dataset(
    dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
    dataset_version="202406",
    variables=["sob"],
    minimum_longitude=64,
    maximum_longitude=91,
    minimum_latitude=2,
    maximum_latitude=26,
    start_datetime="2022-06-05",
    end_datetime="2022-06-10",
    minimum_depth=0.5,
    maximum_depth=0.5,
)

# Temporal mean
def temporal_mean(da):
    for d in ["time", "Time", "date"]:
        if d in da.dims:
            da = da.mean(d, skipna=True)
    return da

sss = temporal_mean(dataset1["sss"])
sst = temporal_mean(dataset2["sst_anomaly"])
ssh = temporal_mean(dataset3["SLA"])
winspd = temporal_mean(dataset4["winspd"])
forecast_sss = temporal_mean(forecast_sss["sob"])

# Rename coords
ssh = ssh.rename({'Latitude': 'lat', 'Longitude': 'lon'})
forecast_sss = forecast_sss.rename({'latitude': 'lat', 'longitude': 'lon'})

# Interpolation
deg_4km = 4.0 / 111.0
target_lat = np.arange(2, 26 + deg_4km, deg_4km)
target_lon = np.arange(64, 91 + deg_4km, deg_4km)

sss = sss.rename({'latitude': 'lat', 'longitude': 'lon'})

sss = sss.interp(lat=target_lat, lon=target_lon)
sst = sst.interp(lat=target_lat, lon=target_lon)
ssh = ssh.interp(lat=target_lat, lon=target_lon)
winspd = winspd.interp(lat=target_lat, lon=target_lon)
forecast_sss = forecast_sss.interp(lat=target_lat, lon=target_lon)

# Prepare arrays
aux = np.stack([sst.values, ssh.values, winspd.values], axis=-1)
aux = np.nan_to_num(aux)
salinity = np.nan_to_num(sss.values)

def minmax(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

salinity = minmax(salinity)
aux[...,0] = minmax(aux[...,0])
aux[...,1] = minmax(aux[...,1])
aux[...,2] = minmax(aux[...,2])

# Low resolution simulation
low_res = tf.image.resize(salinity[np.newaxis, ..., np.newaxis],
                          [salinity.shape[0]//6, salinity.shape[1]//6],
                          method='bicubic').numpy().squeeze()

low_res = tf.image.resize(low_res[np.newaxis, ..., np.newaxis],
                          [salinity.shape[0], salinity.shape[1]],
                          method='bicubic').numpy().squeeze()

# Model
def build_generator(input_shape=(None, None, 4)):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, 12, padding='same')(inputs)
    x = PReLU()(x)
    res = x

    for _ in range(12):
        skip = x
        x = Conv2D(64, 4, padding='same')(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Conv2D(64, 4, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([skip, x])

    x = Conv2D(64, 4, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([res, x])

    x = Conv2D(1, 9, padding='same', activation='sigmoid')(x)
    return Model(inputs, x)

# Build + predict 
generator = build_generator((None, None, 4))

input_tensor = np.stack([
    low_res,
    aux[...,0],
    aux[...,1],
    aux[...,2],
], axis=-1)

super_res = generator.predict(input_tensor[np.newaxis, ...]).squeeze()

# Visualization
# Convert to float32 as scipy.ndimage
super_res_float32 = super_res.astype(np.float32)
fig, axs = plt.subplots(1, 3, figsize=(15,5))
axs[0].imshow(low_res, cmap='viridis', origin='lower'); axs[0].set_title("Low-Resolution Input")
axs[1].imshow(super_res_float32, cmap='viridis', origin='lower'); axs[1].set_title("SRGAN Output")
axs[2].imshow(salinity, cmap='viridis', origin='lower'); axs[2].set_title("True High-Resolution Salinity")
for ax in axs: ax.axis('off')
plt.tight_layout()
plt.show()

# Statistically evaluate model accuracy
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# Ensure all arrays are float32 for consistent calculations
salinity_float = salinity.astype(np.float32)
super_res_float = super_res.astype(np.float32)
low_res_float = low_res.astype(np.float32)
# MSE
mse_low_res = mean_squared_error(salinity_float.flatten(), low_res_float.flatten())
mse_super_res = mean_squared_error(salinity_float.flatten(), super_res_float.flatten())
# PSNR
psnr_low_res = peak_signal_noise_ratio(salinity_float, low_res_float, data_range=1.0)
psnr_super_res = peak_signal_noise_ratio(salinity_float, super_res_float, data_range=1.0)
# SSIM 
ssim_low_res = structural_similarity(salinity_float, low_res_float, data_range=1.0)
ssim_super_res = structural_similarity(salinity_float, super_res_float, data_range=1.0)
# Visualize the difference maps
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(np.abs(salinity_float - low_res_float), cmap='hot', origin='lower')
axs[0].set_title('Absolute Difference: Low-Res vs. True')
axs[0].axis('off')
axs[1].imshow(np.abs(salinity_float - super_res_float), cmap='hot', origin='lower')
axs[1].set_title('Absolute Difference: SRGAN vs. True')
axs[1].axis('off')
plt.tight_layout()
plt.show()

#Spectral Power Comparison
from scipy.fft import fft2, fftshift
# Low-resolution
fft_low_res = fft2(low_res_float)
shifted_fft_low_res = fftshift(fft_low_res)
power_spectrum_low_res = np.abs(shifted_fft_low_res)**2
# Super-resolved
fft_super_res = fft2(super_res_float)
shifted_fft_super_res = fftshift(fft_super_res)
power_spectrum_super_res = np.abs(shifted_fft_super_res)**2
# True high-resolution
fft_salinity = fft2(salinity_float)
shifted_fft_salinity = fftshift(fft_salinity)
power_spectrum_salinity = np.abs(shifted_fft_salinity)**2
# Create a figure with three subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

im0 = axs[0].imshow(np.log(power_spectrum_low_res + 1e-10), cmap='viridis', origin='lower')
axs[0].set_title('Low-Resolution Power Spectrum')
axs[0].axis('off')
fig.colorbar(im0, ax=axs[0], orientation='vertical', label='log(Power)')

im1 = axs[1].imshow(np.log(power_spectrum_super_res + 1e-10), cmap='viridis', origin='lower')
axs[1].set_title('SRGAN Power Spectrum')
axs[1].axis('off')
fig.colorbar(im1, ax=axs[1], orientation='vertical', label='log(Power)')

im2 = axs[2].imshow(np.log(power_spectrum_salinity + 1e-10), cmap='viridis', origin='lower')
axs[2].set_title('True High-Resolution Power Spectrum')
axs[2].axis('off')
fig.colorbar(im2, ax=axs[2], orientation='vertical', label='log(Power)')

plt.tight_layout()
plt.show()

