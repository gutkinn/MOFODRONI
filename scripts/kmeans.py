import numpy as np
from sklearn.cluster import MiniBatchKMeans
from osgeo import gdal
import pandas as pd
import os
import rasterio
from pathlib import Path

def cluster_KMeans(filtered_cropped_file, invalid_pixels_mask, cube_to_cluster_file, output_file, n_clusters, flag_harmonics, batch_size=10000):

    # Create dekadal stacks containing 20 years of observations per pixel
    if not os.path.isfile(cube_to_cluster_file):
        print("Creating Statistics Cube first")
        get_statistics_cube(filtered_cropped_file, invalid_pixels_mask, cube_to_cluster_file)
        
    if flag_harmonics:
        
        harmonics_cube_path = invalid_pixels_mask.replace("invalid_pixel_stack.tif","harmonics_cube.tif")
        if not os.path.isfile(harmonics_cube_path):
            print("Creating Harmonics Cube")
            get_harmonic_cube(invalid_pixels_mask,harmonics_cube_path)
        harmonics_cube = rasterio.open(harmonics_cube_path).read()
        harmonics_cube[np.where(harmonics_cube==0)]=np.nan
        for band in range(harmonics_cube.shape[0]):
            harmonics_cube[band,:,:] = np.interp(harmonics_cube[band], (np.nanmin(harmonics_cube[band]),
                                                                        np.nanmax(harmonics_cube[band])), (-1, +1))
        harmonics_cube = np.nan_to_num(harmonics_cube, nan=-9999)
    
        harmonics_transposed = np.transpose(harmonics_cube,(1,2,0))     
    
    dataset = gdal.Open(cube_to_cluster_file)
    
    # Extract band names from metadata
    metadata = dataset.GetMetadata()
    band_names = [metadata.get(f'Band_{i+1}', f'Timestep_{i+1}') for i in range(dataset.RasterCount)]
    if flag_harmonics:
        band_names+=['a0','amplitude1','phase1','amplitude2','phase2','amplitude3','phase3']

    # Read the bands into a numpy array
    bands = []
    for i in range(dataset.RasterCount):
        band = dataset.GetRasterBand(i + 1)
        bands.append(band.ReadAsArray())

    # Stack and reshape bands into a 2D array
    bands_2d = np.dstack(bands)
    print(f'bands_2 shape: {bands_2d.shape}')
    if flag_harmonics:
        bands_combined = np.concatenate((bands_2d, harmonics_transposed), axis=2) 
    else:
        bands_combined = bands_2d
        
    data = bands_combined.reshape(-1, bands_combined.shape[2])
    
    print("Performing KMeans Clustering")
    #kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, max_iter=1000, max_no_improvement=100, reassignment_ratio=0) VICI
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, max_iter=1000)
    kmeans.fit(data)

    # Get the cluster labels

    labels = kmeans.labels_
    zone_labels = np.unique(labels)

    remap_zones = {zone:list(zone_labels).index(zone) for zone in zone_labels}

    for key in remap_zones.keys():
        labels[labels==key] = remap_zones[key]

    # Reshape labels back to the original image shape
    clustered_image = labels.reshape(bands_2d.shape[:2])

    # Save the clustered image as a .tif file
    driver = gdal.GetDriverByName('GTiff')
    output_dataset = driver.Create(output_file, bands_2d.shape[1], bands_2d.shape[0], 1, gdal.GDT_Int32)

    # Copy geotransform, projection, and metadata from the input dataset
    output_dataset.SetGeoTransform(dataset.GetGeoTransform())
    output_dataset.SetProjection(dataset.GetProjection())
    output_dataset.SetMetadata(dataset.GetMetadata())

    output_dataset.GetRasterBand(1).WriteArray(clustered_image)
    output_dataset.FlushCache()
    output_dataset = None
    dataset = None
    
    centers_csv = output_file.replace(".tif",".csv")
    
    # Save cluster centers as CSV
    cluster_centers = kmeans.cluster_centers_
    df_centers = pd.DataFrame(cluster_centers, columns=band_names)
    df_centers.index.name = "Cluster"
    df_centers.to_csv(centers_csv)

    print(f"Cluster centers saved to {centers_csv}")

def get_harmonic_cube(raster_stack, output_file, harmonics = 3):
    
    # Configuration
    n_dekads_per_year = 36

    # Read data
    with rasterio.open(raster_stack) as src:
        NDVI_cube = src.read()  # shape: (time, x, y)
        profile = src.profile

    t_size, x_size, y_size = NDVI_cube.shape
    n_timesteps = t_size
    print(f"Cube shape: {x_size} x {y_size} x {t_size}")

    # Build time vector and design matrix X
    t = np.arange(n_timesteps) / n_dekads_per_year  # time in years
    omega = 2 * np.pi  # annual cycle

    X = [np.ones_like(t)]
    for k in range(1, harmonics + 1):
        X.append(np.cos(k * omega * t))
        X.append(np.sin(k * omega * t))
    X = np.vstack(X).T  # shape: (t, 2*harmonics + 1)
    n_coeffs = X.shape[1]

    # Output coefficient and derived maps
    a0 = np.zeros((x_size, y_size), dtype=np.float32)
    amplitudes = [np.zeros((x_size, y_size), dtype=np.float32) for _ in range(harmonics)]
    phases = [np.zeros((x_size, y_size), dtype=np.float32) for _ in range(harmonics)]

    # Loop over each pixel
    for i in tqdm(range(x_size),desc="iterating rows"):
        for j in range(y_size):
            y_pixel = NDVI_cube[:, i, j]  # time series for this pixel

            if np.all(np.isnan(y_pixel)):
                continue  # skip if all values are NaN

            # Optionally: mask NaNs (if partial time series exists)
            valid_mask = ~np.isnan(y_pixel)
            if valid_mask.sum() < n_coeffs:
                continue  # not enough data to fit

            X_valid = X[valid_mask]
            y_valid = y_pixel[valid_mask]

            # Least squares fit
            coeffs, _, _, _ = np.linalg.lstsq(X_valid, y_valid, rcond=None)

            a0[i, j] = coeffs[0]
            for h in range(harmonics):
                a = coeffs[1 + 2*h]
                b = coeffs[2 + 2*h]
                amplitudes[h][i, j] = np.sqrt(a**2 + b**2)
                phases[h][i, j] = np.arctan2(-b, a)

    # Stack result layers
    result_layers = [a0]
    for amp, phase in zip(amplitudes, phases):
        result_layers.append(amp)
        result_layers.append(phase)

    results = np.stack(result_layers, axis=0)  # shape: (7, x, y)

    # Write output raster
    profile.update(count=7, dtype="float32")
    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(results.astype(np.float32))
    
    
def get_statistics_cube(filtered_cropped_file, invalid_pixels_mask, cube_to_cluster_file):
    '''
    Compute percentiles p10, p50, p90, and the standard deviation of data stored
    in filtered_cropped_file per dekad and combine all into one datacube that
    will be used in the KMeans clustering.
    Pixels in invalid_pixels_mask and pixels with more than 17/20 years of invalid data are masked.

    Parameters
    ----------
    filtered_cropped_file : str
        File with upper envelope filtered and cropped data cube.
    invalid_pixels_mask : 2D array
        Array with invalid pixel mask.
    cube_to_cluster_file : str
        Filename for statistics datacube.

    Returns
    -------
    None (datacube containing masked p10, p50, p90, and std values is saved).
    '''
    print('Building cube of stats to be used in KMeans clustering...')
    
    with rasterio.open(filtered_cropped_file) as src:
        profile = src.profile
        data = src.read()
        
    with rasterio.open(invalid_pixels_mask) as src:
        invalid_data = src.read()

    soildata_dir = Path(cube_to_cluster_file).parent.parent.parent
    with rasterio.open(os.path.join(soildata_dir,'soildata','soc030_reproj.tif')) as src:
        soc = src.read().astype(np.float32)
    with rasterio.open(os.path.join(soildata_dir,'soildata','clay05_reproj.tif')) as src:
        clay05 = src.read(1).astype(np.float32)
    with rasterio.open(os.path.join(soildata_dir,'soildata','clay515_reproj.tif')) as src:
        clay515 = src.read(1).astype(np.float32)
    with rasterio.open(os.path.join(soildata_dir,'soildata','clay1530_reproj.tif')) as src:
        clay1530 = src.read(1).astype(np.float32)

    dekads, nx, ny = data.shape

    # Initialize storage arrays
    p10_year = np.full((36, nx, ny), np.nan, dtype=np.float32)
    p50_year = np.full((36, nx, ny), np.nan, dtype=np.float32)
    p90_year = np.full((36, nx, ny), np.nan, dtype=np.float32)
    std_year = np.full((36, nx, ny), np.nan, dtype=np.float32)
    lta_year = np.full((36, nx, ny), np.nan, dtype=np.float32)
    lon_year = np.full((1, nx, ny), np.nan, dtype=np.float32)
    lat_year = np.full((1, nx, ny), np.nan, dtype=np.float32)
    soc_year = np.full((1, nx, ny), np.nan, dtype=np.float32)
    clay05_year = np.full((1, nx, ny), np.nan, dtype=np.float32)
    clay515_year = np.full((1, nx, ny), np.nan, dtype=np.float32)
    clay1530_year = np.full((1, nx, ny), np.nan, dtype=np.float32)

    for i in range(36):
        print(f'Dekad {i}', end='\r', flush=True)
        dekad_bands = np.arange(i, dekads, 36)
        cube = data[dekad_bands, :, :].astype(float)
        cube_inv = invalid_data[dekad_bands,:,:]

        # Mask 0 values as NaN (assuming 0 is nodata)
        cube[cube == 0] = np.nan

        # Skip processing if the entire slice is NaN
        if np.all(np.isnan(cube)):
            print(f"Skipping dekad {i} as it contains only NaNs")
            continue

        # Check if more than 17/20 years are invalid
        dekad_mask = np.count_nonzero(~np.isnan(cube), axis=0) <= 1
        invalid_pixels_mask = np.count_nonzero(~np.isnan(cube_inv),axis=0) <= 1

        # Compute percentiles and standard deviation
        p10 = np.percentile(cube, 10, axis=0)
        p50 = np.percentile(cube, 50, axis=0)
        p90 = np.percentile(cube, 90, axis=0)
        std = np.std(cube, axis=0)
        lta = np.mean(cube, axis=0)
        indices = np.dstack(np.indices(p10.shape))

        # Apply invalid pixel mask and dekad mask
        total_mask = invalid_pixels_mask.astype(bool) | dekad_mask.astype(bool)
        p10[total_mask] = np.nan
        p50[total_mask] = np.nan
        p90[total_mask] = np.nan
        std[total_mask] = np.nan
        lta[total_mask] = np.nan
        
        #Normalize the columns
        p10_normalized = np.interp(p10, (np.nanmin(p10), np.nanmax(p10)), (-1, +1))
        p50_normalized = np.interp(p50, (np.nanmin(p50), np.nanmax(p50)), (-1,+1))
        p90_normalized = np.interp(p90, (np.nanmin(p90), np.nanmax(p90)), (-1,+1))
        std_normalized = np.interp(std,(np.nanmin(std), np.nanmax(std)), (-1,+1))
        lta_normalized = np.interp(lta, (np.nanmin(lta), np.nanmax(lta)), (-1,+1))

        p10_year[i, :, :] = p10_normalized
        p50_year[i, :, :] = p50_normalized
        p90_year[i, :, :] = p90_normalized
        std_year[i, :, :] = std_normalized
        lta_year[i, :, :] = lta_normalized
        
    lat = indices[:,:,0].astype(np.float32)
    lon = indices[:,:,1].astype(np.float32)    
    lat_normalized = np.interp(lat, (np.nanmin(lat), np.nanmax(lat)), (-1,+1))
    lon_normalized = np.interp(lon, (np.nanmin(lon), np.nanmax(lon)), (-1,+1))

    soc_normalized = np.interp(soc, (np.nanmin(soc), np.nanmax(soc)), (-1,+1))
    clay05_normalized = np.interp(clay05, (np.nanmin(clay05), np.nanmax(clay05)), (-1,+1))
    clay515_normalized = np.interp(clay515, (np.nanmin(clay515), np.nanmax(clay515)), (-1,+1))
    clay1530_normalized = np.interp(clay1530, (np.nanmin(clay1530), np.nanmax(clay1530)), (-1,+1))
    
    lat_year[0, :, :] = lat_normalized
    lon_year[0, :, :] = lon_normalized
    soc_year[0, :, :] = soc_normalized
    clay05_year[0, :, :] = clay05_normalized
    clay515_year[0, :, :] = clay515_normalized
    clay1530_year[0, :, :] = clay1530_normalized

    # Assemble the input cube for clustering
    cube_to_cluster = np.concatenate((p10_year, p50_year, p90_year, std_year, lta_year,
                                      lat_year, lon_year, soc_year, clay05_year, clay515_year, clay1530_year), axis=0)
    
    # Convert NaN to 0 for compatibility with the clustering algorithm
    cube_to_cluster = np.nan_to_num(cube_to_cluster, nan=-9999)

    # Update profile for multi-band output
    profile.update({'dtype': 'float32', 'count': 186})

    with rasterio.open(cube_to_cluster_file, 'w', **profile) as dst:
        dst.write(cube_to_cluster.astype(rasterio.float32))

    print(f"Statistics cube saved as: {cube_to_cluster_file}")