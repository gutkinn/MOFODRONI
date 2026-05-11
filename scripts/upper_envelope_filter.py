#!/usr/bin/env python3
# This script can only be run on Windows as it requires the Ilwis package.
# To install ilwispy download the Windows wheel for 20230428 from https://filetransfer.itc.nl/pub/52n/ilwis_py/wheels/windows/
# and install in a conda environment of python 3.8 via: python -m pip install <path to the Windows wheel>
import os
import glob
import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import ilwis
from osgeo import gdal
import rasterio
import subprocess

from centaurvici.util import common_functions


def extent_timeseries(input_folder, start_date, end_date):
    '''
    Extent temporal period with 2 months on both ends if possible.
    If no extra data is available at start or end of temporal range,
    don't extent range on that side.

    Parameters
    ----------
    input_folder : str
        Folder with NDVI data as .tif
    start_date : str
        Start date of temporal extent
    end_date : str
        End date of temporal extent

    Returns
    -------
    new_filelist : list
        List with filenames of data included in the extended time period
    skip_dekads_start : int
        Number of additional dekads at start of extended time period that
        should be removed after upper envelope filtering
    '''
    filelist = glob.glob(f'{input_folder}/*.tif')
    filelist.sort()

    start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    start_date_max_extended = (start_datetime - relativedelta(months=2))
    start_date_max_extended_file = input_folder + '\\' + str(start_date_max_extended.strftime('%Y-%m-%d')) + '.tif'

    end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    end_date_max_extended = (end_datetime + relativedelta(months=2))
    end_date_max_extended_file = input_folder + '\\' + str(end_date_max_extended.strftime('%Y-%m-%d')) + '.tif'

    if start_date_max_extended_file in filelist:
        start_date_extended = start_date_max_extended
        skip_dekads_start = 6
    else:
        start_date_extended = start_datetime
        skip_dekads_start = 0
        print('!!! Warning: no upper envelope filtering for first 2 months of temporal period !!!')

    if end_date_max_extended_file in filelist:
        end_date_extended = end_date_max_extended
    else:
        end_date_extended = end_datetime
        print('!!! Warning: no upper envelope filtering for last 2 months of temporal period !!!')

    new_filelist = []
    for file in filelist:
        date = datetime.datetime.strptime(file.split('\\')[-1].split('.tif')[0], '%Y-%m-%d')
        if (date >= start_date_extended) and (date <= end_date_extended):
            new_filelist.append(file)
    new_filelist.sort()

    start_date_extended = start_date_extended.strftime('%Y-%m-%d')
    end_date_extended = end_date_extended.strftime('%Y-%m-%d')
    return new_filelist, start_date_extended, end_date_extended, skip_dekads_start


def makeMaplist(input_folder, start_date, end_date):
    '''
    Determine the extended temporal range for the upper envelope filtering
    and combine all data into one raster.

    Parameters
    ----------
    input_folder : str
        Folder with NDVI data as .tif files
    start_date : str
        Start date of temporal extent
    end_date : str
        End date of temporal extent

    Returns
    -------
    maplist : 3D raster
        All NDVI data within extended time period
    skip_dekads_start : int
        Number of additional dekads at start of extended time period that
        should be removed after upper envelope filtering
    '''
    # Create list of files used in upper envelope filtering
    filelist, start_date_extended, end_date_extended, skip_dekads_start = extent_timeseries(input_folder, start_date,
                                                                                            end_date)
    print(f'Number of dekads in extended timeseries: {len(filelist)}')

    # Use first file to determine geo reference
    file = filelist[0]
    ds = rasterio.open(file)
    data = ds.read(1)
    size = np.shape(data)

    lonmin = ds.transform[2]
    latmax = ds.transform[5]
    lonmax = lonmin + size[1] * ds.transform[0]
    latmin = latmax + size[0] * ds.transform[4]
    envelope = str(lonmin) + ' ' + str(latmax) + ' ' + str(lonmax) + ' ' + str(latmin)
    ds.close()
    # print(f'Columns-Rows and the Lat-Long window-extent WNES: \n {str(size[1]) + ' ' + str(size[0])} \n {envelope}')

    grf = ilwis.GeoReference('code=georef:type=corners,csy=epsg:4326,envelope=' + envelope +
                             ',gridsize=' + str(size[1]) + ' ' + str(size[0]) +
                             ',cornerofcorners=yes,name=filtered')
    dfNum = ilwis.DataDefinition(ilwis.NumericDomain("code=value"), ilwis.NumericRange(0, 255, 1))
    maplist = ilwis.RasterCoverage()
    maplist.setSize(ilwis.Size(size[1], size[0], len(filelist)))
    maplist.setGeoReference(grf)
    maplist.setDataDef(dfNum)

    # Group all files into one raster
    all_layers = []
    for file in filelist:
        with rasterio.open(file) as ds:
            data = ds.read(1)
            all_layers.append(data)

    ds = None
    all_layers = np.array(all_layers)
    maplist.array2raster(all_layers)
    return maplist, start_date_extended, end_date_extended, skip_dekads_start


#############################################################################
#############################################################################
def upper_envelope_filtering(directory, start_date, end_date, shapefile):
    '''
    Determine extended time period and perform upper envelope filtering on
    all NDVI data within that period. Remove the added dekads on both ends
    and cut the data to the borders of the country.

    Parameters
    ----------
    directory : str
        Name of the input/output directory for the country, containing
        the 'NDVI_downloaded' folder
    start_date : str
        Start date of temporal extent
    end_date : str
        End date of temporal extent
    shapefile : str
        Path to the country shapefile

    Returns
    -------
    None
    '''
    # Set input directory and name of output file/folders
    input_NDVI_downloaded = f'{directory}/NDVI_downloaded'
    filtered_folder = f'{directory}/filtered'
    if not os.path.isdir(filtered_folder):
        os.makedirs(filtered_folder)
    filtered_cropped_folder = f'{directory}/filtered_cropped'
    if not os.path.isdir(filtered_cropped_folder):
        os.makedirs(filtered_cropped_folder)

    # ----------------------------------------------------------------
    print('Generating maplist')
    ndvi, start_date_extended, end_date_extended, skip_dekads_start = makeMaplist(input_NDVI_downloaded, start_date,
                                                                                  end_date)
    out_filtered = f'{directory}/filtered_{start_date_extended}_{end_date_extended}.img'

    # ----------------------------------------------------------------
    # Set all values above 250 and all Nans to zero
    print('Removing quality flag')
    ndvi = ilwis.do('mapcalc', 'iff(@1>250,0,@1)', ndvi)
    ndvi = ilwis.do('mapcalc', 'iff(@1==?,0,@1)', ndvi)

    # ----------------------------------------------------------------
    print('Performing timesat filter')
    # Arguments in timesat are: input grid, #iterations with different window size (if 4,
    # window sizes of 3, 5, 7 and 9 are used), upper envelop only, upper envelope
    # in last iteration if False, extend window at beginning and end of timeseries
    # More info here: https://ftp.itc.nl/pub/52n/EENSAT/notebooks/notebook_results/Intro_timeseries_ILWISPy_result.html
    filtered = ilwis.do('timesat', ndvi, 4, True, True, False)
    filtered.store(out_filtered, 'HFA', 'gdal')
    ndvi = None

    # ----------------------------------------------------------------
    print('Cutting to correct time period')
    t = gdal.Open(out_filtered)
    data = t.ReadAsArray()

    all_dates = pd.date_range(start_date, end_date, freq='D')
    dates = []
    for d in all_dates:
        if d.day in [1, 11, 21]:
            dates.append(d.strftime('%Y-%m-%d'))

    # Create new raster
    print('Number of dekads in filtered timeseries: ', len(dates))
    for bix in range(len(dates)):
        filename = f'{filtered_folder}/{dates[bix]}.tif'
        if not os.path.isfile(filename):
            data_dekad = data[bix + skip_dekads_start, :, :]
            data_dekad[data_dekad > 250] = 250  # Due to filtering values might be above 250 -> max is 250 (NDVI = 0.92)
            common_functions.writeGeotiff(data_dekad, filename,
                                          t.GetProjection(), t.GetGeoTransform(),
                                          nodataValue=0, datatype='uint8')

            # Crop the data to only cover the country
            filename_cropped = f'{filtered_cropped_folder}/{dates[bix]}.tif'
            command = f'gdalwarp -cutline {shapefile} -srcnodata 0 -dstnodata 0 {filename} {filename_cropped}'
            subprocess.call(command, shell=True)

    return


if __name__ == '__main__':
    # Inputs
    country = 'Somalia'
    directory = f'/vitodata/CENTAUR/data/vici/Sarah/{country}'
    start_date = '1999-01-01'  # Day = start of a dekad (1, 11, or 21)
    end_date = '2020-06-21'  # Day = start of a dekad (1, 11, or 21)
    shapefile = f'/home/sarahg/centaur_vici/src/centaurvici/resources/AOIs-Africa/{country}_25km.shp'

    # Run the script
    upper_envelope_filtering(directory, start_date, end_date, shapefile)