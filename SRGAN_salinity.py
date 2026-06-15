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

# Enable Mixed Precision (for T4 Tensor Core acceleration)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision policy:", mixed_precision.global_policy())

# Imports
import earthaccess
import os
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import rioxarray
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Input, Conv2D, PReLU, BatchNormalization, Add, LeakyReLU, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import scipy.ndimage
from scipy.fft import fft2, fftshift
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import mean_squared_error
import copernicusmarine
import zarr
import gc

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

# Clear Memory After Interpolation
# Delete unused large xarray datasets
del dataset1, dataset2, dataset3
gc.collect()
print("Memory cleared after interpolation.")

# Prepare arrays
aux = np.stack([sst.values, ssh.values, winspd.values], axis=-1)
aux = np.nan_to_num(aux)
salinity = np.nan_to_num(sss.values)
forecast_sss = np.nan_to_num(forecast_sss.values)

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

# Define SRGAN Architecture
def build_generator(input_shape=(None, None, 4)):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, 12, padding='same')(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    res = x

    # Residual blocks
    for _ in range(12):
        skip = x
        x = Conv2D(64, 4, padding='same')(x)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(64, 4, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([skip, x])

    x = Conv2D(64, 4, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([res, x])

    x = Conv2D(1, 9, padding='same', activation='sigmoid')(x)
    return Model(inputs, x, name='Generator')

def build_discriminator(input_shape=(None, None, 1)):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, 4, strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)
    for filters in [128, 256, 512]:
        x = Conv2D(filters, 4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    # The last layer should be a Dense layer for classification in a typical discriminator
    x = Dense(1, activation='sigmoid', dtype='float32')(x)
    return Model(inputs, x, name='Discriminator')

print(f"Salinity shape for model building: {salinity.shape}")
generator = build_generator((None, None, 4))
discriminator = build_discriminator((salinity.shape[0], salinity.shape[1], 1))

# Augmentation
@tf.function
def augment(x, y):

    # Random horizontal flip
    flip_lr = tf.random.uniform(()) > 0.5
    x = tf.cond(flip_lr,
                lambda: tf.image.flip_left_right(x),
                lambda: x)
    y = tf.cond(flip_lr,
                lambda: tf.image.flip_left_right(y),
                lambda: y)

    # Random vertical flip
    flip_ud = tf.random.uniform(()) > 0.5
    x = tf.cond(flip_ud,
                lambda: tf.image.flip_up_down(x),
                lambda: x)
    y = tf.cond(flip_ud,
                lambda: tf.image.flip_up_down(y),
                lambda: y)

    # Random 90° rotation
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    x = tf.image.rot90(x, k)
    y = tf.image.rot90(y, k)

    # Random brightness scaling
    scale = tf.random.uniform([],0.95,1.05)
    x = tf.clip_by_value(x*scale,0.0,1.0)

    # Gaussian noise (input only)
    noise = tf.random.normal(tf.shape(x),0.0,0.01)
    x = tf.clip_by_value(x+noise,0.0,1.0)

    return x, y

#Define Gradient Loss
def gradient_loss(y_true, y_pred):
    dy_true = y_true[:,1:,:,:] - y_true[:,:-1,:,:]
    dy_pred = y_pred[:,1:,:,:] - y_pred[:,:-1,:,:]
    dx_true = y_true[:,:,1:,:] - y_true[:,:,:-1,:]
    dx_pred = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:]
    return tf.reduce_mean(tf.abs(dx_true - dx_pred)) + \
           tf.reduce_mean(tf.abs(dy_true - dy_pred))
           
# Define a custom loss function that combines MSE and gradient loss
def generator_total_loss(y_true, y_pred):
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    grad_loss = gradient_loss(y_true, y_pred) 
    return mse_loss + (1e-2 * grad_loss) 
# Cosine Decay Learning Rate Scheduler
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=5e-5,
    decay_steps=800,
    alpha=1e-2
)
opt = Adam(learning_rate=lr_schedule)
discriminator.compile(loss='binary_crossentropy', optimizer=opt)
discriminator.trainable = True

gan_input = Input(shape=(salinity.shape[0], salinity.shape[1], 4))
gen_out = generator(gan_input)
gan_out = discriminator(gen_out)
srgan = Model(gan_input, [gen_out, gan_out])


# Compile SRGAN with two losses: one for generator output, one for discriminator output
srgan.compile(loss=[generator_total_loss, 'binary_crossentropy'],
              loss_weights=[1, 4e-4], # weight for generator_total_loss and binary_crossentropy
              optimizer=opt)
# Freeze ONLY inside SRGAN
discriminator.trainable = False

# Create input tensor: 
input_tensor = np.stack([
    low_res,
    aux[...,0],
    aux[...,1],
    aux[...,2],
], axis=-1)

target_tensor = salinity[..., np.newaxis]
train_dataset = tf.data.Dataset.from_tensors(
    (input_tensor, target_tensor)
)
train_dataset = (
    train_dataset
    .cache()  # Keeps data in RAM (very important)
    .map(lambda x, y: (
            tf.cast(x, tf.float32),
            tf.cast(y, tf.float32)),
         num_parallel_calls=tf.data.AUTOTUNE)
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(1) # drop_remainder=True is not necessary for a single element dataset
    .prefetch(tf.data.AUTOTUNE)
)

# Enable Dataset Optimization
options = tf.data.Options()
options.experimental_optimization.map_parallelization = True
options.autotune.enabled = True # Corrected line
options.experimental_optimization.apply_default_optimizations = True
train_dataset = train_dataset.with_options(options)
tf.config.optimizer.set_jit(True)


# Training loop (GPU + Mixed Precision + tf.data)
full_input_batch = input_tensor[np.newaxis, ...]
full_target_batch = target_tensor[np.newaxis, ...]
best_g_loss=np.inf
patience=100
wait=0

for step in range(800):
    # Using the full tensors directly as a batch of 1
    batch_input = full_input_batch
    batch_target = full_target_batch
    fake_sal = generator(batch_input, training=True)
    d_loss_real = discriminator.train_on_batch(
        batch_target, np.ones((1,1))
    )
    d_loss_fake = discriminator.train_on_batch(
        fake_sal, np.zeros((1,1))
    )
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    g_loss = srgan.train_on_batch(
        batch_input,
        [batch_target, np.ones((1,1))]
    )
    current_loss=float(g_loss[0])

    if current_loss < best_g_loss:
        best_g_loss=current_loss
        wait=0
        generator.save_weights("best_generator.weights.h5")
    else:
        wait+=1

    if step % 100 == 0:
        current_lr=float(opt.learning_rate(step).numpy()) if callable(opt.learning_rate) else float(tf.keras.backend.get_value(opt.learning_rate))
        print(f"Step {step}: LR={current_lr:.2e}, D loss={d_loss:.4f}, G loss={current_loss:.4f}")

    if wait>=patience:
        print(f"Early stopping at step {step}")
        break

generator.load_weights("best_generator.weights.h5")
print("Loaded best generator weights.")

# Visualize Super-Resolution Result
super_res = generator.predict(input_tensor[np.newaxis, ...]).squeeze()
import scipy.ndimage
# Convert to float32 as scipy.ndimage does not support float16 for gaussian_filter
super_res_float32 = super_res.astype(np.float32)
fig, axs = plt.subplots(1, 3, figsize=(15,5))
axs[0].imshow(low_res, cmap='viridis', origin='lower'); axs[0].set_title("Low-Resolution Input")
axs[1].imshow(super_res, cmap='viridis', origin='lower'); axs[1].set_title("SRGAN Output (Super-Resolved)")
axs[2].imshow(forecast_sss, cmap='viridis', origin='lower'); axs[2].set_title("True High-Resolution Salinity")
for ax in axs: ax.axis('off')
plt.tight_layout()
plt.show()

# Statistically evaluate model accuracy
# Ensure all arrays are float32 for consistent calculations
salinity_float = salinity.astype(np.float32)
super_res_float = super_res.astype(np.float32)
low_res_float = low_res.astype(np.float32)
# MSE
mse_low_res = mean_squared_error(salinity_float, low_res_float)
mse_super_res = mean_squared_error(salinity_float, super_res_float)
# PSNR (Requires data range, assuming 0-1 from normalization)
psnr_low_res = peak_signal_noise_ratio(salinity_float, low_res_float, data_range=1.0)
psnr_super_res = peak_signal_noise_ratio(salinity_float, super_res_float, data_range=1.0)
# SSIM (Requires data range, assuming 0-1 from normalization)
# Ensure dimensions are (H, W) or (H, W, C) for SSIM, our data is (H, W)
ssim_low_res = structural_similarity(salinity_float, low_res_float, data_range=1.0)
ssim_super_res = structural_similarity(salinity_float, super_res_float, data_range=1.0)
print("--- Model Accuracy Metrics ---")
print(f"MSE (Low-Res vs. True): {mse_low_res:.4f}")
print(f"MSE (SRGAN vs. True): {mse_super_res:.4f}")
print(f"PSNR (Low-Res vs. True): {psnr_low_res:.2f} dB")
print(f"PSNR (SRGAN vs. True): {psnr_super_res:.2f} dB")
print(f"SSIM (Low-Res vs. True): {ssim_low_res:.4f}")
print(f"SSIM (SRGAN vs. True): {ssim_super_res:.4f}")
# You can also visualize the difference maps
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
# Compute FFT, shift, and power spectrum for each dataset
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
# Plot Low-Resolution Power Spectrum
im0 = axs[0].imshow(np.log(power_spectrum_low_res + 1e-10), cmap='viridis', origin='lower')
axs[0].set_title('Low-Resolution Power Spectrum')
axs[0].axis('off')
fig.colorbar(im0, ax=axs[0], orientation='vertical', label='log(Power)')
# Plot SRGAN Power Spectrum
im1 = axs[1].imshow(np.log(power_spectrum_super_res + 1e-10), cmap='viridis', origin='lower')
axs[1].set_title('SRGAN Power Spectrum')
axs[1].axis('off')
fig.colorbar(im1, ax=axs[1], orientation='vertical', label='log(Power)')
# Plot True High-Resolution Power Spectrum
im2 = axs[2].imshow(np.log(power_spectrum_salinity + 1e-10), cmap='viridis', origin='lower')
axs[2].set_title('True High-Resolution Power Spectrum')
axs[2].axis('off')
fig.colorbar(im2, ax=axs[2], orientation='vertical', label='log(Power)')
plt.tight_layout()
plt.show()


