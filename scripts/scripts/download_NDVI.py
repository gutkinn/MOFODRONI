#!/usr/bin/env python3
# ---------------------------------------------------
# Step 1: Download NDVI data for 20 years
# --> Vici_historical_downloader_v5.py

# see also script test_jeroen.py, first part...

# the script should automatically take care of the resolution,
# depending on the date...
# (instead of having separate functions for historical vs contemporary)

# ideally, upper envelop filtering is done directly in openeo...
# --> this should be tested using script of Dieter
# (VICI_droughts_calculator_v2.py in centaurvici)

# DETAILED STEPS:
# - get correct country shapefile
# - make necessary corrections to AOI (see test_jeroen.py)
# - automatically select the time range (20 years,
# starting from start_date)
# - start openeo job including data download and upper envelop filtering
# - save results to outdir

import os
import glob
import numpy as np
import pandas as pd
import openeo
import xarray as xr
from osgeo import gdal
import subprocess

import centaurvici.util.openeo_udf.get_NOBS_mask
import centaurvici.util.openeo_udf.mask_invalid_values
GET_NOBS_MASK_FILE = centaurvici.util.openeo_udf.get_NOBS_mask.__file__
MASK_INVALID_VALUES_FILE = centaurvici.util.openeo_udf.mask_invalid_values.__file__

from centaurvici.util import common_functions


############################################################################
def get_data_openeo(collection, temporal_extent, spatial_extent, res, outdir):
    '''
    Download NDVI data via OpenEO.

    Parameters
    ----------
    collection : str
        ID of the NDVI collection in OpenEO
    temporal_extent : list
        Start and end date of period that is downloaded
    spatial_extent : dict
        east, west, north, south coordinates of region that is downloaded
    res : str
        Resolution (1km or 300m)
    outdir : str
        Output directory for downloaded NDVI data

    Returns
    -------
    None
    '''
    print(f'Downloading NDVI data for: {collection}, {temporal_extent}, {spatial_extent}, {res}')

    # Authenticate in OpenEO
    eoconn = openeo.connect('openeo.vito.be')
    eoconn.authenticate_oidc()

    # Download NDVI data
    if res == '1km':
        # Load NDVI data
        ds = eoconn.load_collection(
            collection,
            temporal_extent=temporal_extent,
            spatial_extent=spatial_extent,
            bands='NDVI')

    elif res == '300m':
        # Load NDVI data
        ds_NDVI = eoconn.load_collection(
            collection,
            temporal_extent=temporal_extent,
            spatial_extent=spatial_extent,
            bands="NDVI")
        cube_1km = ds_NDVI.resample_spatial(resolution=3.0*0.00297619047619, method="average")

        # Load NOBS data
        ds_NOBS = eoconn.load_collection(
            collection,
            temporal_extent=temporal_extent,
            spatial_extent=spatial_extent,
            bands="NOBS")
        NOBS_1km = ds_NOBS.resample_spatial(resolution=3.0*0.00297619047619, method="average")

        # Mask pixels without observations (NOBS = 0)
        udf_mask_code = openeo.UDF.from_file(GET_NOBS_MASK_FILE)
        NOBS_1km_mask = NOBS_1km.apply(process=udf_mask_code)

        temp_cube = cube_1km.merge_cubes(NOBS_1km_mask)
        temp_cube = temp_cube.rename_labels("bands", ["NDVI","NOBS_flag"])

        ds = (temp_cube.band("NDVI") * temp_cube.band("NOBS_flag")).add_dimension('bands', label='NDVI', type='bands')


    # Mask invalid pixels (DN > 250)
    udf_mask_code = openeo.UDF.from_file(MASK_INVALID_VALUES_FILE)
    ds_masked = ds.apply(process=udf_mask_code)

    job = ds_masked.execute_batch(out_format='GTIFF', title='Download_ndvi')
    results = job.get_results()
    results.get_metadata()
    results.download_files(outdir)

    print(f'Data downloaded to {outdir}')
    return


def get_data_local_folder(AOI_adjusted_file, res, temporal_extent, outdir):
    '''
    Copy NDVI data from local directory for given temporal range and
    crop in to spatial extents given in the AOI_adjusted_file.

    Parameters
    ----------
    AOI_adjusted_file : str
        File with AOI boundaries
    res : str
        Resolution (1km or 300m)
    temporal_extent : list
        Start and end date of period that is downloaded
    outdir : str
        Output directory for copied NDVI data

    Returns
    -------
    None
    '''
    AOI_Adjusted = pd.read_csv(AOI_adjusted_file)

    # Select local folder depending on resolution (1km or 300m)
    if res == '1km':
        NDVI_dir = '/data/MTDA/BIOPAR/BioPar_NDVI_V3_Global'
        rows_cols = AOI_Adjusted[AOI_Adjusted['type'] == 'COP_1k'][['StartCol','StartRow','Columns','Rows']]

    elif res == '300m':
        NDVI_dir = '/data/MTDA/BIOPAR/BioPar_NDVI300_V2_Global'
        rows_cols = AOI_Adjusted[AOI_Adjusted['type'] == 'COP_1k300m'][['StartCol','StartRow','Columns','Rows']]
        new_res = 3.0*0.00297619047619

    # Get spatial extent (which rows and columns)
    xmin = rows_cols['StartCol'].values[0]
    ymin = rows_cols['StartRow'].values[0]
    xmax = rows_cols['StartCol'].values[0] + rows_cols['Columns'].values[0]
    ymax = rows_cols['StartRow'].values[0] + rows_cols['Rows'].values[0]

    # Determine all dates within temporal extent
    all_dates = pd.date_range(temporal_extent[0], temporal_extent[1], freq='D')
    dates = []
    for d in all_dates:
        if d.day in [1,11,21]:
            dates.append(d.strftime('%Y-%m-%d'))

    # Crop data to spatial extent and save in output directory
    for i in range(len(dates)):
        date = dates[i]
        year = date[:4]

        if not os.path.isfile(f'{outdir}/{date}.tif'):
            filename = glob.glob(f'{NDVI_dir}/{year}/{date.replace('-','')}/*/*.nc')[0]
            f = xr.open_dataset(filename, mask_and_scale=False, decode_times=False, engine='netcdf4')

            if res == '1km':
                ndvi = f['NDVI']
                ndvi_data = ndvi[0, ymin:ymax:1,xmin:xmax:1]

                # Mask invalid pixels (DN > 250)
                ndvi_data.values[ndvi_data.values > 250] = 255

                output_file = f'{outdir}/{date}.tif'
                geotransform = common_functions.calculateGeotransformXY(ndvi_data.lon, ndvi_data.lat)
                common_functions.writeGeotiff(ndvi_data.values, output_file, projection=f.crs.attrs['spatial_ref'],
                                              geotransform=geotransform, datatype='uint8', nodataValue=255)

            elif res == '300m':
                # NDVI
                ndvi = f['NDVI']
                ndvi_data = ndvi[0, ymin:ymax:1,xmin:xmax:1]
                output_file_ndvi = f'{outdir}/{date}_ndvi.tif'
                output_file_ndvi_resampled = f'{output_file_ndvi.split('.tif')[0]}_resampled.tif'
                geotransform = common_functions.calculateGeotransformXY(ndvi_data.lon, ndvi_data.lat)
                common_functions.writeGeotiff(ndvi_data.values, output_file_ndvi, projection=f.crs.attrs['spatial_ref'],
                                              geotransform=geotransform, datatype='uint8', nodataValue=255)

                # Resample NDVI to 1 km
                command = f'gdalwarp -tr {new_res} {new_res} -r average {output_file_ndvi} {output_file_ndvi_resampled}'
                subprocess.call(command, shell=True)

                # NOBS
                nobs = f['NOBS']
                nobs_data = nobs[0, ymin:ymax:1,xmin:xmax:1]
                output_file_nobs = f'{outdir}/{date}_nobs.tif'
                output_file_nobs_resampled = f'{output_file_nobs.split('.tif')[0]}_resampled.tif'
                geotransform = common_functions.calculateGeotransformXY(nobs_data.lon, nobs_data.lat)
                common_functions.writeGeotiff(nobs_data.values, output_file_nobs, projection=f.crs.attrs['spatial_ref'],
                                              geotransform=geotransform, datatype='uint8', nodataValue=255)

                # Resample NOBS to 1km
                command = f'gdalwarp -tr {new_res} {new_res} -r average {output_file_nobs} {output_file_nobs_resampled}'
                subprocess.call(command, shell=True)

                f.close()
                ndvi = ndvi_data = nobs = nobs_data = None

                # Load resampled datasets
                ndvi = gdal.Open(output_file_ndvi_resampled)
                ndvi_data = ndvi.ReadAsArray()
                nobs = gdal.Open(output_file_nobs_resampled)
                nobs_data = nobs.ReadAsArray()

                # Mask pixels without observations (NOBS = 0)
                ndvi_data = np.where(nobs_data > 0, ndvi_data, 255)

                # Mask invalid pixels (DN > 250)
                ndvi_data[ndvi_data > 250] = 255

                output_file = f'{outdir}/{date}.tif'
                common_functions.writeGeotiff(ndvi_data, output_file, projection=ndvi.GetProjection(),
                                              geotransform=ndvi.GetGeoTransform(), datatype='uint8', nodataValue=255)

                # Remove intermediate files
                os.remove(output_file_ndvi)
                os.remove(output_file_ndvi_resampled)
                os.remove(output_file_nobs)
                os.remove(output_file_nobs_resampled)

    return

#############################################################################
#############################################################################
def download_ndvi_data(country, output_directory, start_date, end_date, use_openeo):
    '''
    Determine spatial extent for the given country and download NDVI data
    using OpenEO or copy from local directory.

    Parameters
    ----------
    country : str
        Name of the country
    output_directory : str
        Name of the output directory
    start_date : str
        Start date of the temporal range
    end_date: str
        End date of the temporal range
    use_openeo : boolean
        If true download from OpenEO, otherwise copy data from NDVI_dir

    Returns
    -------
    None (NDVI data is saved in the output_directory)
    '''
    print(f'Downloading NDVI data for {country} ...')
    # Set up the structure of the output directory
    output_folder = f'{output_directory}/{country}'
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    output_folder_NDVI = f'{output_directory}/{country}/NDVI_downloaded'
    if not os.path.isdir(output_folder_NDVI):
        os.makedirs(output_folder_NDVI)

    # Temporal extent
    temporal_extent_1km, temporal_extent_300m = common_functions.get_temporal_extent(start_date, end_date)

    # Spatial extent
    AOI_adjusted_file = f'{output_folder}/AOI_adjusted.csv'
    spatial_extent_1km, spatial_extent_300m = common_functions.get_spatial_extent(country, AOI_adjusted_file)

    # Download NDVI data (openEO or copy from local folder)
    if temporal_extent_1km:
        res = '1km'
        collection_id = 'CGLS_NDVI_V3_GLOBAL'
        if use_openeo:
            # Start openeo job including data download and upper envelop filtering
            get_data_openeo(collection_id, temporal_extent_1km, spatial_extent_1km, res, output_folder_NDVI)
        else:
            # Copy data from local folder
            get_data_local_folder(AOI_adjusted_file, res, temporal_extent_1km, output_folder_NDVI)

    if temporal_extent_300m:
        res = '300m'
        collection_id = 'CGLS_NDVI300_V2_GLOBAL'
        if use_openeo:
            # Start openeo job including data download and upper envelop filtering
            get_data_openeo(collection_id, temporal_extent_300m, spatial_extent_300m, res, output_folder_NDVI)
        else:
            # Copy data from local folder
            get_data_local_folder(AOI_adjusted_file, res, temporal_extent_300m, output_folder_NDVI)

    return



if __name__ == '__main__':
    # Inputs
    country = 'Ethiopia'
    output_directory = '/vitodata/CENTAUR/data/vici/Sarah'
    start_date = '2021-01-01'
    end_date = '2021-01-21'
    use_openeo = True

    # Run the script
    download_ndvi_data(country, output_directory, start_date, end_date, use_openeo=use_openeo)