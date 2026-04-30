#!/usr/bin/env python3
import sys
import os
from loguru import logger
import pandas as pd
import numpy as np
import csv
from osgeo import gdal
import rasterio
import rioxarray as rxr
from tqdm import tqdm
from scipy import stats
import datetime
import rsgislib
import rsgislib.imagecalc
import rsgislib.rastergis
import rsgislib.segmentation
from pathlib import Path

#This won't work as of now as this is an internal package. It is only for writing the geotiff. Rewrite for rasterio.
#from centaurvici.util import common_functions

#os.environ['PROJ_LIB'] = '/home/sarahg/anaconda3/envs/cent_vici/share/proj'

current_dir = os.path.dirname(__file__)
resources_folder = os.path.join(os.path.split(current_dir)[0], 'resources')


#############################################################################
### Functions for step 1
#############################################################################
def step1_invalid_pixel_mask(filtered_cropped_file, invalid_pixels_mask_file):
    '''
    Determine historically invalid pixels based on long term trends.

    Parameters
    ----------
    filtered_cropped_file : str
        File with upper envelope filtered and cropped data cube
    invalid_pixels_mask_file : str
        Filename of invalid pixels mask

    Returns
    -------
    invalid_pixel_mask : 2D array
        Array with invalid pixel mask, also saved in country_dir/invalid_pixels_historical.tif
    '''
    print('Building historical map of invalid pixels...')

    t = gdal.Open(filtered_cropped_file)
    data = t.ReadAsArray()
    ndates, nx, ny = data.shape

    # Mask water pixels
    water_mask = np.mean(data, axis=0) == 0

    # Mask pixels with extreme long term trends
    data = np.reshape(data, (ndates, nx * ny))
    x = np.arange(len(data))
    slope_all_years = np.zeros(nx * ny)
    for i in tqdm(range(nx * ny)):
        pixel = data[:, i].astype(float)
        flag = pixel == 0  # nodatavalue is 0
        pixel = pixel[flag == False]

        if len(pixel) > 0:
            x1 = x[flag == False]
            slope = stats.linregress(x1, pixel).slope
            slope_all_years[i] = slope
        else:
            slope_all_years[i] = 1

    slope_all_years = np.reshape(slope_all_years, [nx, ny])
    invalid_all_years = np.array(slope_all_years < -0.084883047) | np.array(slope_all_years > 0.0814077858)

    # Combine water mask and long term trends mask into one masked pixel file
    invalid_pixels_mask = invalid_all_years | water_mask

    common_functions.writeGeotiff(invalid_pixels_mask, invalid_pixels_mask_file,
                                  projection=t.GetProjection(), geotransform=t.GetGeoTransform(),
                                  datatype='uint8', nodataValue=255)
    t = None
    return invalid_pixels_mask


#############################################################################
### Functions for step 2
#############################################################################
import numpy as np
import rasterio
from tqdm import tqdm

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
    will be used in the IsoData clustering.
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
    print('Building cube of stats to be used in IsoData clustering...')
    
    with rasterio.open(filtered_cropped_file) as src:
        profile = src.profile
        data = src.read()
        
    with rasterio.open(invalid_pixels_mask) as src:
        invalid_data = src.read()

    with rasterio.open(r'/home/eoafrica/shared/MOFODRONI/soil_data/soc030_reproj_sm.tif') as src:
        soc = src.read().astype(np.float32)
    with rasterio.open(r'/home/eoafrica/shared/MOFODRONI/soil_data/clay05_reproj_sm.tif') as src:
        clay05 = src.read(1).astype(np.float32)
    with rasterio.open(r'/home/eoafrica/shared/MOFODRONI/soil_data/clay515_reproj_sm.tif') as src:
        clay515 = src.read(1).astype(np.float32)
    with rasterio.open(r'/home/eoafrica/shared/MOFODRONI/soil_data/clay1530_reproj_sm.tif') as src:
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



def rearrange_gmtxt_file(centr, centrX):
    '''
    Transform the tabular data in the input file 'centr' such that
    the columns are arranged in sorted order. The order key is determined
    by the average of the values in each column.

    Parameters
    ----------
    centr : str
        Input file
    centrX : str
        Output file

    Returns
    -------
    None
    '''
    # Read the lines with #classes and #bands
    head = []
    with open(centr, 'r') as file:
        head.append(file.readline())
        head.append(file.readline())

    # Read the tabular data
    df = pd.read_csv(centr, header=None, skiprows=2)

    # Calculate mean per column and sort the aggregates values
    # the sort retains the original columns positions in the table
    key = df.mean(axis=0).sort_values()

    # Use the list of original columns positions as index to rearange the columns
    dfs = df.iloc[:, key.index]

    # Write the lines with the #classes and #bands, and write the tabular data
    with open(centrX, 'w') as file:
        file.writelines(head)
        s = dfs.to_csv(header=False, index=False)
        file.write(s)
    return


def determine_clusters(cube_to_cluster_file, CPSZ_file, min_zones, max_zones, sub_sample, max_std):
    '''
    Perform clustering algorithm to determine between a number of
    min_zones and max_zones clusters.

    Parameters
    ----------
    cube_to_cluster_file : str
        File with statistics datacube
    CPSZ_file : str
        Filename for CPS zones
    min_zones : int
        Minimal number of zones that may be determined
    max_zones : int
        Maximal number of zones that may be determined
    sub_sample : int
        Free parameter in clustering algorithm
    max_std : int
        Free parameter in clustering algorithm

    Returns
    -------
    None (clustering is saved in country_dir/CPSZ.tif)
    '''
    kea_img = f'{cube_to_cluster_file.split('.img')[0]}.kea'
    out_kea_img = f'{cube_to_cluster_file.split('.img')[0]}_ISO.kea'
    centr = f'{cube_to_cluster_file.split('.img')[0]}_legend.gmtxt'
    centrX = f'{cube_to_cluster_file.split('.img')[0]}_legend_sorted.gmtxt'
    out_img = f'{cube_to_cluster_file.split('.img')[0]}_ISO.img'

    print("Input image:     " + cube_to_cluster_file)
    print("Centroids file:  " + centr)
    
    print("Opening Stack, this can take a while- be patient :)")
    
    kea_options = gdal.TranslateOptions(format="KEA")
    
    ds = gdal.Open(cube_to_cluster_file)
    ds = gdal.Translate(kea_img, ds, options=kea_options)
    ds = None

    print('Clustering...')
    # max_std seems to be the parameter that most influences the output
    list_of_parameters = [(max_std, sub_sample)]

    done = False
    nruns = 1
    while not done:
        rsgislib.imagecalc.isodata_clustering(
            kea_img, centr,
            n_clusters=100,  # The default was 10
            max_n_iters=200,
            # The smaller, the longer (exponentially) it takes and way more clusters are identified.
            # For a TM, the default is 100 (a 1% sub-sample).
            # If more bands are included [hyper-temporal], then the sub-sample must become larger!
            sub_sample=sub_sample,
            ignore_zeros=True,
            degree_change=0.0025,
            init_cluster_method=rsgislib.INITCLUSTER_DIAGONAL_FULL_ATTACH,
            min_dist_clusters=10,  # default 1, erdas uses 4
            min_n_feats=200,
            max_std_dev=max_std,  # default 30, erdas uses 5, when set higher, the number of classes quickly decreases
            min_n_clusters=10,
            start_iter=10,
            end_iter=100,
        )

        rearrange_gmtxt_file(centr, centrX)
        rsgislib.segmentation.label_pixels_from_cluster_centres(kea_img, out_kea_img, centrX,
                                                                ignore_zeros=True, gdalformat="KEA")
        rsgislib.rastergis.pop_rat_img_stats(clumps_img=out_kea_img, add_clr_tab=True,
                                             calc_pyramids=True, ignore_zero=True)
        
        print("Translating, this might take a while: relax!")

        ds = gdal.Open(out_kea_img)
        ds = gdal.Translate(out_img, ds, format='HFA')
        ds = None

        class_map = gdal.Open(out_kea_img).ReadAsArray()
        n_classes = len(np.unique(class_map))
        print(f'A total of {n_classes - 1} classes were detected.')

        if n_classes < min_zones:
            if max_std >= 5:
                max_std -= 1
            else:
                sub_sample -= 1
            list_of_parameters.append((max_std, sub_sample))
            print(list_of_parameters)

            nruns += 1
            if nruns > 100:
                print('Could not fit the number of zones in the given range. Exiting...')
                print(f'Tried the following combinations (max_std, sub_sample): {list_of_parameters[:-1]}')
                sys.exit('Stopped here. Check max_std and sub_sample ranges.')

        elif n_classes > max_zones:
            if sub_sample <= 9:
                sub_sample += 1
            else:
                max_std += 1
            list_of_parameters.append((max_std, sub_sample))
            print(list_of_parameters)

            nruns += 1
            if nruns > 100:
                print('Could not fit the number of zones in the given range. Exiting...')
                print(f'Tried the following combinations (max_std, sub_sample): {list_of_parameters[:-1]}')
                sys.exit('Stopped here. Check max_std and sub_sample ranges.')
        else:
            print('Successfully fitted the number of zones in the given range.')
            print(f'Tried the following combinations (max_std, sub_sample): {list_of_parameters}')
            done = True

    # Make it .tif to use in the automated processing
    t = gdal.Open(out_kea_img)
    data = t.ReadAsArray()
    profile = {
        "driver": "GTiff",
        "dtype": "int16",
        "nodata": 32767,
        "width": t.RasterXSize,
        "height": t.RasterYSize,
        "count": 1,  # Assuming a single-band raster
        "crs": t.GetProjection(),
        "transform": t.GetGeoTransform()
    }
    with rasterio.open(CPSZ_file, "w", **profile) as dst:
        dst.write(data, 1)  # Write data to band 1
    t = None
    return


#############################################################################
def step2_isodata_clustering(filtered_cropped_file, invalid_pixels_mask,
                             cube_to_cluster_file, CPSZ_file, min_zones,
                             max_zones, sub_sample, max_std):
    '''
    Compute statistics on dekadal stacks and perform isodata clustering.

    Parameters
    ----------
    filtered_cropped_file : str
        File with upper envelope filtered and cropped data cube
    invalid_pixels_mask : 2D array
        Array with invalid pixels mask
    cube_to_cluster_file : str
        Filename for statistics cube
    CPSZ_file : str
        Filename for CPS zones
    min_zones : int
        Minimal number of zones that may be determined
    max_zones : int
        Maximal number of zones that may be etermined
    sub_sample : int
        Free parameter in clustering algorithm
    max_std : int
        Free parameter in clustering algorithm

    Returns
    -------
    None
    '''
    print('Computing statistics on dekadal stacks and performing isodata clustering...')

    # Create dekadal stacks containing 20 years of observations per pixel
    if not os.path.isfile(cube_to_cluster_file):
        get_statistics_cube(filtered_cropped_file, invalid_pixels_mask, cube_to_cluster_file)

    # Run the isodata clustering algorithm
    determine_clusters(cube_to_cluster_file, CPSZ_file, min_zones, max_zones, sub_sample, max_std)
    return


#############################################################################
### Functions for step 3
#############################################################################
def export_adjustments_by_dekad(zone_adjust_dir, C_Adj_file):
    '''
    Save NDVI adjustments data per dekad.

    Parameters
    ----------
    zone_adjust_dir : str
        Output directory for zonal adjustments
    C_Adj_file : str
        Filename of datacube with zonal adjustments
    Returns
    -------
    None (results are saved in folder country_dir/adjustments/zone_adjust/)
    '''
    days = ["01", "11", "21"]
    months = [month for month in range(1, 13)]
    dekads = [f'{b:0>2d}{a}' for b in months for a in days]

    t = gdal.Open(C_Adj_file)
    data = t.ReadAsArray()
    for i in range(len(dekads)):
        im = data[i, :, :]
        file = f'{zone_adjust_dir}/zone_adjust_{dekads[i]}.mpr.tif'

        # Create new raster
        common_functions.writeGeotiff(im, file, projection=t.GetProjection(),
                                      geotransform=t.GetGeoTransform(), datatype='int16', nodataValue=32767)

    t = None
    return


#############################################################################
def step3_compute_adjustments(filtered_cropped_file, CPSZ_file, C_Adj_file, outdir_zone_adjust):
    '''
    Compute NDVI adjustments per CPS zone and save the adjustments per dekad.

    Parameters
    ----------
    filtered_cropped_file : str
        File with upper envelope filtered and cropped data cube
    CPSZ_file : str
        File with CPS zones
    C_Adj_file : str
        Filename for zone adjustments
    outdir_zone_adjust: str
        Output directory for zone adjustments per dekad

    Returns
    -------
    None (zone adjustments are saved in country_dir/adjustments/C_Adj.tif,
            and per dekad in country_dir/adjustments/zone_adjust/1970/)
    '''
    print('Computing NDVI adjustments per CPS zone...')
    t = gdal.Open(filtered_cropped_file)
    data = t.ReadAsArray()

    # Read CPSZs
    CPSZs = gdal.Open(CPSZ_file).ReadAsArray()
    zones = np.unique(CPSZs)[1:]  # First zone contains invalid pixels

    # We need the map of pixel means (36 dekads) and the map of zonal medians (36 dekads)
    dekads, nx, ny = data.shape
    pixel_means = np.zeros((36, nx, ny))
    zonal_medians = np.zeros((36, nx, ny)) * np.nan
    for i in range(36):
        print('Dekad ' + str(i), end='\r')
        dekad_bands = np.arange(i, dekads, 36)
        cube = data[dekad_bands, :, :].astype(float)
        cube[cube == 0] = np.nan  # Don't include 0 values in stats computation (0 is used as nodatavalue)

        pixel_means[i, :, :] = np.nanmean(cube, axis=0)

        for zone in zones:
            mask = np.tile(CPSZs == zone, (20, 1, 1)).astype(float)
            mask[mask == 0] = np.nan
            temp_cube = cube * mask
            zonal_medians[i, CPSZs == zone] = np.nanmedian(temp_cube)

    # Invalid pixel mask is included via CPSZ -> zonal_medians -> adjustment_cube
    adjustment_cube = np.round(pixel_means - zonal_medians)
    adjustment_cube[np.isnan(adjustment_cube)] = 32767
    adjustment_cube = adjustment_cube.astype(int)

    # Create new raster
    common_functions.writeGeotiff(adjustment_cube, C_Adj_file,
                                  projection=t.GetProjection(), geotransform=t.GetGeoTransform(),
                                  datatype='int16', nodataValue=32767)
    t = None

    # Save adjustments per dekad
    export_adjustments_by_dekad(outdir_zone_adjust, C_Adj_file)
    return


#############################################################################
### Functions for step 4
#############################################################################
def apply_correction_factor(filtered_cropped_file, C_Adj_file, Adjusted_NDVI_file):
    '''
    Apply the zonal adjustments to the NDVI data.

    Parameters
    ----------
    filtered_cropped_file : str
        File with upper envelope filtered and cropped data cube
    C_Adj_file : str
        File with zonal adjustments per dekad
    Adjusted_NDVI_file : str
        Filename for adjusted NDVI values

    Returns
    -------
    None (datacube with adjusted NDVI values is saved in country_dir/adjustments/Adjusted_NDVI.tif)
    '''
    print('Applying zonal adjustment factor to NDVI values...')
    t = gdal.Open(filtered_cropped_file)
    data = t.ReadAsArray()
    projection = t.GetProjection()
    geotransform = t.GetGeoTransform()

    adj = gdal.Open(C_Adj_file).ReadAsArray()

    NDVI_adjusted = np.zeros(np.shape(data))
    for i in range(t.RasterCount):
        data_dekad = data[i, :, :].astype(float)
        data_dekad[data_dekad == 0] = np.nan
        adj_dekad = adj[np.mod(i, 36), :, :].astype(float)
        adj_dekad[adj_dekad == 32767] = np.nan
        NDVI_adjusted[i, :, :] = data_dekad - adj_dekad
    t = data = adj = data_dekad = adj_dekad = None

    NDVI_adjusted[NDVI_adjusted < 0] = 0  # 0 is smallest possible value (NDVI = -0.08)
    NDVI_adjusted[NDVI_adjusted > 250] = 250  # 250 is largest possible value (NDVI = 0.92)
    NDVI_adjusted[np.isnan(NDVI_adjusted)] = 255

    # Store adjusted cube
    common_functions.writeGeotiff(NDVI_adjusted, Adjusted_NDVI_file,
                                  projection=projection, geotransform=geotransform,
                                  datatype='uint8', nodataValue=255)
    NDVI_adjusted = None
    return


def compute_percentiles(Adjusted_NDVI_file, CPSZ_file, percentile_numbers, outdir_percentiles):
    '''
    Compute dekadal percentiles per CPS zone using the adjusted NDVI data.

    Parameters
    ----------
    Adjusted_NDVI_file : str
        File with adjusted NDVI values
    CPSZ_file : str
        File with CPS zones
    percentile_numbers : list
        List of integers for which percentile should be computed (example: [5,15] -> p5 and p15)
    outdir_percentiles : str
        Directory to save percentiles

    Returns
    -------
    percentiles : dict
        Percentiles (given in percentile_numbers) per dekad and per CPS zone
    p50_array : array
        p50 percentile per dekad and per CPS zone
    '''
    print('Computing dekadal percentiles per CPS zone...')
    t = gdal.Open(Adjusted_NDVI_file)
    data = t.ReadAsArray()
    ndekads, nx, ny = data.shape

    days = ["01", "11", "21"]
    months = [month for month in range(1, 13)]
    dekads = [f'{b:0>2d}{a}' for b in months for a in days]

    # Load CPSZs
    CPSZs = gdal.Open(CPSZ_file).ReadAsArray()
    zones = np.unique(CPSZs)[1:]
    nzones = len(zones)

    # Get percentiles arrays
    percentiles = {}
    for p in percentile_numbers:
        percentiles[f'p{p:02}'] = np.full((36, nx, ny), 255).astype(int)
    p50_array = np.zeros((36, nzones))

    for i in range(36):
        print('Dekad ' + str(i), end='\r')
        dekad_bands = np.arange(i, ndekads, 36)
        cube = data[dekad_bands, :, :].astype(float)
        cube[cube == 255] = np.nan  # No additional masking needed because this was already done in Adjusted_NDVI.tif

        for zone in zones:
            for p in percentile_numbers:
                perc = np.nanpercentile(cube[:, CPSZs == zone], q=p)  # TODO: ok to use nanpercentile?
                if np.isnan(perc):
                    perc = 255
                percentiles[f'p{p:02}'][i, CPSZs == zone] = int(np.round(perc))

            perc50 = np.nanpercentile(cube[:, CPSZs == zone], q=50)  # TODO: ok to use nanpercentile?
            p50_array[i, int(zone) - 1] = perc50

        # Output percentiles files
        for p in percentile_numbers:
            im = percentiles[f'p{p:02}'][i, :, :]
            file = f'{outdir_percentiles}/percentiles{p}_{dekads[i]}.tif'
            common_functions.writeGeotiff(im, file,
                                          projection=t.GetProjection(), geotransform=t.GetGeoTransform(),
                                          datatype='uint8', nodataValue=255)

            # Save p50_array in csv file
    file = f'{outdir_percentiles}/p50_array.csv'
    with open(file, 'w') as f:
        csvWriter = csv.writer(f)
        csvWriter.writerow(zones)
        csvWriter.writerows(p50_array)

    t = CPSZs = None
    return percentiles, p50_array


#############################################################################
def step4_compute_payout_thresholds(filtered_cropped_file, C_Adj_file, Adjusted_NDVI_file,
                                    CPSZ_file, percentile_numbers, outdir_percentiles):
    '''
    Compute the payout thresholds, i.e. the limiting percentiles of adjusted NDVI values
    per dekad and per CPS zone.

    Parameters
    ----------
    filtered_cropped_file : str
        File with upper envelope filtered and cropped data cube
    C_Adj_file : str
        File with zonal adjustments per dekad
    Adjusted_NDVI_file : str
        Filename for adjusted NDVI values
    CPSZ_file : str
        File with CPS zones
    percentile_numbers : list
        List of integers for which percentile should be computed (example: [5,15] -> p5 and p15)
    outdir_percentiles : str
        Directory to save percentiles

    Returns
    -------
    percentiles : dict
        Percentiles (given in percentile_numbers) per dekad and per CPS zone
    p50_array : 2D array
        p50 percentile per dekad and per CPS zone
    '''
    print('Computing the dekadal percentiles per CPS zone using adjusted NDVI data...')

    if not os.path.isfile(Adjusted_NDVI_file):
        apply_correction_factor(filtered_cropped_file, C_Adj_file, Adjusted_NDVI_file)

    percentiles, p50_array = compute_percentiles(Adjusted_NDVI_file, CPSZ_file, percentile_numbers, outdir_percentiles)
    return percentiles, p50_array


#############################################################################
### Functions for step 5
#############################################################################
def step5_get_seasons(p50_array, zones, seasons_file):
    '''
    Determine growth seasons per CPS zone from median NDVI values.

    Parameters
    ----------
    p50_array : 2D array
        p50 percentile per dekad and per CPS zone
    zones : list
        List with zone numbers
    seasons_file : str
        Filename for seasons

    Returns
    -------
    seasons : 2D array
        Boolean array with growth seasons per CPS zone
    '''
    print('Computing seasons...')

    def moving_average(x):
        w = np.ones(9)
        return np.convolve(x, w, 'valid') / 9

    ndekads, nzones = p50_array.shape

    seasons = np.zeros((ndekads, nzones))
    NDVI_thr = 70
    ratio_thr = 0.95
    for i in range(nzones):
        zone_profile = p50_array[:, i]
        max_val = np.max(zone_profile)
        zone_profile2 = np.tile(zone_profile, 2)  # Repeat data
        ma9 = moving_average(zone_profile2)
        zone_profile = zone_profile2[36:]
        ma9 = ma9[28:]
        flag_50perc_ma9 = zone_profile > ma9
        flag_50perc_NDVIthr = zone_profile > NDVI_thr
        flag_50perc_maxval = zone_profile / max_val > ratio_thr

        flag_season = (flag_50perc_ma9 & flag_50perc_NDVIthr) | (
                flag_50perc_maxval & flag_50perc_NDVIthr)

        seasons[:, i] = np.array(flag_season).astype(int)  # 1 = in season, 0 = out of season

    # Write seasons to file
    with open(seasons_file, 'w') as f:
        csvWriter = csv.writer(f)
        csvWriter.writerow(zones)
        csvWriter.writerows(seasons)

    return seasons


#############################################################################
### Functions for step 6
#############################################################################
def step6_correct_percentiles_for_seasons(CPSZ_file, Adjusted_NDVI_file, seasons,
                                          percentiles, percentile_numbers, outdir_percentiles):
    '''
    Correct percentiles for seasonality (mask when out of season) and save them per dekad.

    Parameters
    ----------
    CPSZ_file : str
        File with CPS zones
    Adjusted_NDVI_file : str
        File with adjusted NDVI values
    seasons : 2D array
        Boolean array with growth seasons per CPS zone
    percentiles : dict
        Percentiles (given in percentile_numbers) per dekad and per CPS zone
    percentile_numbers : list
        List of integers for which percentile should be computed (example: [5,15] -> p5 and p15)
    outdir_percentiles : str
        Output directory for percentiles per dekad

    Returns
    -------
    None (percentiles per dekad are saved in country_dir/percentiles/1970/)
    '''
    print('Correcting percentiles for seasons and saving them per dekad and CPS zone...')

    days = ["01", "11", "21"]
    months = [month for month in range(1, 13)]
    dekads = [f'{b:0>2d}{a}' for b in months for a in days]

    # Load CPSZs
    CPSZs = gdal.Open(CPSZ_file).ReadAsArray()
    zones = np.unique(CPSZs)[1:]
    nzones = len(zones)

    # Take geometry and projection from Adjusted_NDVI.tif file
    t = gdal.Open(Adjusted_NDVI_file)

    for i in range(len(dekads)):
        print('Dekad ' + str(i), end='\r')
        # Set percentile arrays to 255 (nodatavalue) when the zone is outside season
        for j in range(nzones):
            mask = CPSZs == zones[j]
            season_flag = seasons[i, j]
            if season_flag == 0:
                for p in percentile_numbers:
                    percentiles[f'p{p:02}'][i, mask == True] = 255

        # Output percentiles files
        for p in percentile_numbers:
            im = percentiles[f'p{p:02}'][i, :, :]

            file = f'{outdir_percentiles}/percentiles{p}_{dekads[i]}.tif'
            common_functions.writeGeotiff(im, file,
                                          projection=t.GetProjection(), geotransform=t.GetGeoTransform(),
                                          datatype='uint8', nodataValue=255)

    return


#############################################################################
#############################################################################
def prepare_vici_country(country, outdir, start_date, end_date, **parameters):
    '''
    Prepare historical VICI inputs for a country.
    Determine invalid pixels, get zones via clustering algorithm,
    compute NDVI adjustments and percentiles, adjusted for seasons, per zone.

    Parameters
    ----------
    country : str
        Country name
    outdir : str
        Output directory
    start_date : str
        Start date of temporal period
    end_date : str
        End date of temporal period
    **parameters : dict
        Additional parameters

    Returns
    -------
    None (all output is saved in subdirectories of outdir)
    '''
    logger.info(f'Preparing VICI inputs for {country}...')
    logger.info(f'Output directory: {outdir}')
    logger.info(f'Parameters: {parameters}')

    # Set up the structure of the output directory
    archive_dir = f'{outdir}/Archive'
    if not os.path.isdir(archive_dir):
        os.mkdir(archive_dir)

    outdir_clustering = f'{archive_dir}/clustering'
    outdir_adjustments = f'{archive_dir}/adjustments'
    outdir_percentiles = f'{archive_dir}/percentiles/1970'
    outdir_percentiles_seasons = f'{archive_dir}/percentiles_seasons/1970'

    filtered_cropped_file = f'{archive_dir}/filtered_cropped_{start_date}_{end_date}.tif'
    invalid_pixels_mask_file = f'{archive_dir}/invalid_pixels_mask.tif'
    cube_to_cluster_file = f'{outdir_clustering}/cube_to_cluster.img'
    CPSZ_file = f'{outdir_clustering}/CPSZ.tif'
    C_Adj_file = f'{outdir_adjustments}/C_Adj.tif'
    Adjusted_NDVI_file = f'{outdir_adjustments}/Adjusted_NDVI.tif'
    seasons_file = f'{archive_dir}/seasons.csv'

    # ---------------------------------------------------
    # Step 0: Combine upper envelope filtered NDVI data
    # for the given temporal range into one datacube file
    print('--------------------------- STEP 0 ---------------------------')
    if not os.path.isfile(filtered_cropped_file):
        start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        all_filtered_data = []
        for file in sorted(os.listdir(f'{outdir}/filtered_cropped')):
            date = datetime.datetime.strptime(file.split('.tif')[0], '%Y-%m-%d')
            if (date >= start_datetime) and (date <= end_datetime):
                t = gdal.Open(os.path.join(f'{outdir}/filtered_cropped', file))
                data = t.ReadAsArray()
                all_filtered_data.append(data)

        common_functions.writeGeotiff(all_filtered_data, filtered_cropped_file,
                                      projection=t.GetProjection(), geotransform=t.GetGeoTransform(),
                                      datatype='uint8', nodataValue=0)
        t = None

    # ---------------------------------------------------
    # Step 1: Create mask of invalid pixels based on downloaded data
    # --> Vici_historical_downloader_v5.py
    # Question: does the current routine also take into account the following
    # condition?
    # pixel/dekad is invalid if more than 17/20 observations invalid
    # NOTE that this should be checked prior to the upper envelop filtering?
    # NOTE THAT THE NEW VERSION OF VICI NO LONGER CONSIDERS SHORT
    # TERM TRENDS AS CRITERION FOR INVALID PIXEL MASKING!
    # Masked pixels should be ignored during future analysis!
    print('--------------------------- STEP 1 ---------------------------')
    if not os.path.isfile(invalid_pixels_mask_file):
        invalid_pixels_mask = step1_invalid_pixel_mask(filtered_cropped_file, invalid_pixels_mask_file)
    else:
        t = gdal.Open(invalid_pixels_mask_file)
        invalid_pixels_mask = t.ReadAsArray()

    # ---------------------------------------------------
    # Step 2: ISODATA clustering

    # Daniel's script: Vici_historical_processor_v1.py
    # --> according to the manual, clustering is done on percentile stacks.
    # Daniel may have used the original data instead??

    # DETAILED STEPS:
    # take the NDVI data and create 36 (36 dekads) stacks of dekadal data,
    # each stack containing 20 observations for each pixel (20 years)
    # in case more than 12 consecutive dekads have no data,
    # upper envelope automatically sets those to zero (to be checked?)
    # now compute the percentiles and SD for each stack.
    # IMPORTANT: DON'T INCLUDE ZEROS WHEN COMPUTING PERCENTILES!!!
    # apply the valid pixel mask!!
    # After generating the individual p10, p50, p90 and sd stacks,
    # then combine them again into one individual file per dekad.
    # These percentile stacks should then be fed into the isodata clustering...
    print('--------------------------- STEP 2 ---------------------------')
    if not os.path.isdir(outdir_clustering):
        os.mkdir(outdir_clustering)
        min_zones = parameters['min_zones'] if 'min_zones' in parameters else 80
        max_zones = parameters['max_zones'] if 'max_zones' in parameters else 120
        sub_sample = parameters['sub_sample'] if 'sub_sample' in parameters else 10
        max_std = parameters['max_std'] if 'max_std' in parameters else 8

        step2_isodata_clustering(filtered_cropped_file, invalid_pixels_mask,
                                 cube_to_cluster_file, CPSZ_file, min_zones,
                                 max_zones, sub_sample, max_std)

    # ---------------------------------------------------
    # Step 3: compute adjustment factors per zone
    # Once you have the zones, you create, for each zone,
    # the 10, 50, 90 percentiles and std statistics.
    # You use particularly the 50 percentiles to compute the adjustment factor for each pixel in each zone.
    print('--------------------------- STEP 3 ---------------------------')
    if not os.path.isdir(outdir_adjustments):
        os.mkdir(outdir_adjustments)
        outdir_zone_adjust = f'{outdir_adjustments}/zone_adjust/1970'
        os.makedirs(outdir_zone_adjust)
        step3_compute_adjustments(filtered_cropped_file, CPSZ_file, C_Adj_file, outdir_zone_adjust)

    # ---------------------------------------------------
    # Step 4: apply correction factor to get corrected 20 year archive
    # and compute pay-out thresholds (p5 and p15 percentiles).
    # You apply the correction factor on all the 20 year data,
    # because you need the CORRECTED data in order to compute
    # the 5 and 15 percentiles (pay-out thresholds)
    # IMPORTANT: in order to compute the pay-out thresholds,
    # you use ALL pixels for ALL years in a certain zone
    # -> combine into one pile of data before you compute the percentiles.
    # IMPORTANT: ignore zero values again!!
    print('--------------------------- STEP 4 ---------------------------')
    percentile_numbers = [5, 15]
    if 'percentile_numbers' in parameters:
        percentile_numbers = parameters['percentile_numbers']

    if not os.path.isdir(outdir_percentiles):
        os.makedirs(outdir_percentiles)
        percentiles, p50_array = step4_compute_payout_thresholds(filtered_cropped_file, C_Adj_file,
                                                                 Adjusted_NDVI_file, CPSZ_file,
                                                                 percentile_numbers, outdir_percentiles)
    else:
        p50_file = f'{outdir_percentiles}/p50_array.csv'
        p50_array = pd.read_csv(p50_file).to_numpy().astype(int)

        percentiles = {}
        for p in percentile_numbers:
            perc_array = []
            for file in sorted(os.listdir(outdir_percentiles)):
                if file.startswith(f'percentiles{p}'):
                    data = rxr.open_rasterio(os.path.join(outdir_percentiles, file))
                    perc_array.extend(data.values)
            percentiles[f'p{p:02}'] = np.array(perc_array)

    # ---------------------------------------------------
    # Step 5: compute growing seasons for each zone
    # In order to compute the growing seasons,
    # you start from the 50 percentile files PER ZONE.
    # (the same you used for computing the correction factors)
    print('--------------------------- STEP 5 ---------------------------')
    CPSZs = gdal.Open(CPSZ_file).ReadAsArray()
    zones = np.unique(CPSZs)[1:]
    seasons = step5_get_seasons(p50_array, zones, seasons_file)

    # ---------------------------------------------------
    # Step 6: correct percentiles for seasonality
    print('--------------------------- STEP 6 ---------------------------')
    if not os.path.isdir(outdir_percentiles_seasons):
        os.makedirs(outdir_percentiles_seasons)
        step6_correct_percentiles_for_seasons(CPSZ_file, Adjusted_NDVI_file, seasons, percentiles,
                                              percentile_numbers, outdir_percentiles_seasons)

    return


if __name__ == '__main__':
    # Mandatory inputs
    country = 'Ethiopia'
    output_directory = f'/vitodata/CENTAUR/data/vici/Sarah/{country}'
    start_date = '2000-01-01'
    end_date = '2019-12-21'

    # Optional inputs
    # min_zones = minimum number of zones in clustering (default=80)
    # max_zones = maximum number of zones in clustering (default=120, Somalia=120, Mali=120, Mozambique=130, Ethiopia=120)
    # sub_sample = free parameter in clustering (Somalia=10, Mali=12, Mozambique=15, Ethiopia=10)
    # max_std = free parameter in clustering (Somalia=8, Mali=4, Mozambique=13, Ethiopia=8)
    # percentiles = list of percentiles that are computed (default = [5,15])
    parameters = {'min_zones': 80, 'max_zones': 120, 'sub_sample': 10, 'max_std': 8, 'percentile_numbers': [5, 15]}

    # Run the script
    prepare_vici_country(country, output_directory, start_date, end_date, **parameters)