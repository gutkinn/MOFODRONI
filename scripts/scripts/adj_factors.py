import os
import rasterio
import numpy as np
import csv

def export_adjustments_by_dekad(zone_adjust_dir, C_adj_file):
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
    os.makedirs(zone_adjust_dir,exist_ok=True)
    
    with rasterio.open(C_adj_file,'r') as src:
        profile = src.profile
        data = src.read()
    profile['count']=1
    
    for i in range(len(dekads)):
        im = data[i, :, :]
        file = f'{zone_adjust_dir}/zone_adjust_{dekads[i]}.tif'

        with rasterio.open(file,'w',**profile) as dst:
            dst.write(im,1)

    return

def compute_adjustments(filtered_cropped_file, CPSZ_file, out_adj_file, outdir):

    print('Computing NDVI adjustments per CPS zone...')
    with rasterio.open(filtered_cropped_file,'r') as src:
        profile = src.profile
        data = src.read()
        tags = src.tags()
    
    # Read CPSZs
    CPSZs = rasterio.open(CPSZ_file).read()
    zones = np.unique(CPSZs)
    
    dekads, nx, ny = data.shape
    pixel_means = np.zeros((36, nx, ny))
    zonal_medians = np.zeros((36, nx, ny)) * np.nan

    for i in range(36):
        print('Dekad ' + str(i), end='\r')
        dekad_bands = np.arange(i, dekads, 36)
        cube = data[dekad_bands, :, :].astype(float)
        
        cube[cube == 0] = np.nan  # Don't include 0 values in stats computation (0 is used as nodatavalue)
    
        pixel_means[i, :, :] = np.nanmean(cube, axis=0)
        num_years = len(dekad_bands)

        for zone in zones:
            mask = np.tile(CPSZs == zone, (num_years, 1, 1)).astype(float)
            mask[mask == 0] = np.nan
            temp_cube = cube * mask
            zonal_medians[i, (CPSZs == zone)[0]] = np.nanmedian(temp_cube)

    # Invalid pixel mask is included via CPSZ -> zonal_medians -> adjustment_cube
    adjustment_cube = np.round(pixel_means - zonal_medians)
    adjustment_cube[np.isnan(adjustment_cube)] = 32767
    adjustment_cube = adjustment_cube.astype(int)

    profile['nodata'] = 32767
    profile['dtype'] = 'int16'
    profile['count']=adjustment_cube.shape[0]
    
    # Create new raster
    with rasterio.open(out_adj_file,'w',**profile) as dst:
        dst.write(adjustment_cube)
        #dst.update_tags(tags)

    # Save adjustments per dekad
    print('Saving dekadal adjustments')
    outdir_zone_adjust = os.path.join(outdir,'adjustments')
    export_adjustments_by_dekad(outdir_zone_adjust, out_adj_file)
    
    print('Finished saving zonal adjustments.')
    return 
    
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
    with rasterio.open(filtered_cropped_file,'r') as src:
        profile = src.profile
        data = src.read()
        tags = src.tags()
    
    with rasterio.open(C_Adj_file,'r') as src:
        adj = src.read()
        
    NDVI_adjusted = np.zeros(np.shape(data))
    for i in range(profile['count']):
        data_dekad = data[i, :, :].astype(float)
        data_dekad[data_dekad == 0] = np.nan
        adj_dekad = adj[np.mod(i, 36), :, :].astype(float)
        adj_dekad[adj_dekad == 32767] = np.nan
        NDVI_adjusted[i, :, :] = data_dekad - adj_dekad
    
    NDVI_adjusted[NDVI_adjusted < 0] = 0  # 0 is smallest possible value (NDVI = -0.08)
    NDVI_adjusted[NDVI_adjusted > 250] = 250  # 250 is largest possible value (NDVI = 0.92)
    NDVI_adjusted[np.isnan(NDVI_adjusted)] = 255

    profile['nodata'] = 255
    with rasterio.open(Adjusted_NDVI_file,'w',**profile) as dst:
        dst.write(NDVI_adjusted)
        dst.update_tags(**tags)
        
    return 
    
def compute_percentiles(Adjusted_NDVI_file, CPSZ_file, percentile_numbers,outdir):
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
    with rasterio.open(Adjusted_NDVI_file,'r') as src:
        profile=src.profile
        data = src.read()
        
    profile['dtype'] = 'uint8'
    profile['nodata'] = 255
    profile['count']=1
    ndekads, nx, ny = data.shape

    days = ["01", "11", "21"]
    months = [month for month in range(1, 13)]
    dekads = [f'{b:0>2d}{a}' for b in months for a in days]

    # Load CPSZs
    CPSZs = rasterio.open(CPSZ_file).read(1)
    zones = np.unique(CPSZs)
    nzones = len(zones)

    # Get percentiles arrays
    percentiles = {}
    for p in percentile_numbers:
        p = int(p)
        percentiles[f'p{p:02}'] = np.full((36, nx, ny), 255).astype(int)
    p50_array = np.zeros((36, nzones))
    
    lower_per = os.makedirs(os.path.join(outdir,'p'+percentile_numbers[0]),exist_ok=True)
    upper_per = os.makedirs(os.path.join(outdir,'p'+percentile_numbers[1]),exist_ok=True)
    
    for i in range(36):
        print('Dekad ' + str(i), end='\r')
        dekad_bands = np.arange(i, ndekads, 36)
        cube = data[dekad_bands, :, :].astype(float)
        cube[cube == 255] = np.nan  # No additional masking needed because this was already done in Adjusted_NDVI.tif

        for zone in zones:

            for p in percentile_numbers:
                p = int(p)
                perc = np.nanpercentile(cube[:, CPSZs == zone], q=p)  # TODO: ok to use nanpercentile?
                if np.isnan(perc):
                    perc = 255
                percentiles[f'p{p:02}'][i, CPSZs == zone] = int(np.round(perc))

            perc50 = np.nanpercentile(cube[:, CPSZs == zone], q=50)  # TODO: ok to use nanpercentile?
            
            p50_array[i, int(zone)] = perc50
        
        # Output percentiles files
        for p in percentile_numbers:
            p = int(p)
            im = percentiles[f'p{p:02}'][i, :, :]
            file = f'{outdir}/p{p:02}/ndvi{p:02}_{dekads[i]}.tif'
            
            with rasterio.open(file,'w',**profile) as dst:
                dst.write(im,1)

    # Save p50_array in csv file
    file = f'{outdir}/p50_array.csv'
    with open(file, 'w') as f:
        csvWriter = csv.writer(f)
        csvWriter.writerow(zones)
        csvWriter.writerows(p50_array)

    return percentiles, p50_array

def get_seasons(p50_array, zones, seasons_file):
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
    print('Finished computing seasons.')
    
    return seasons

def correct_percentiles_for_seasons(CPSZ_file,seasons,percentiles, percentile_numbers, outdir):
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
    lower_per = os.path.join(outdir,'p'+percentile_numbers[0])
    upper_per = os.path.join(outdir,'p'+percentile_numbers[1])
    
    days = ["01", "11", "21"]
    months = [month for month in range(1, 13)]
    dekads = [f'{b:0>2d}{a}' for b in months for a in days]

    # Load CPSZs
    with rasterio.open(CPSZ_file,'r') as src:
        CPSZs = src.read(1)
        profile = src.profile
        
    zones = np.unique(CPSZs)
    nzones = len(zones)
    
    profile['dtype'] = 'uint8'
    profile['nodata'] = 255
    
    for i in range(len(dekads)):
        print('Dekad ' + str(i), end='\r')
        # Set percentile arrays to 255 (nodatavalue) when the zone is outside season
        for j in range(nzones):
            mask = CPSZs == zones[j]
            season_flag = seasons[i, j]
            if season_flag == 0:
                for p in percentile_numbers:
                    p=int(p)
                    percentiles[f'p{p:02}'][i, mask == True] = 253

        # Output percentiles files
        for p in percentile_numbers:
            p = int(p)
            im = percentiles[f'p{p:02}'][i, :, :]

            file = f'{outdir}/p{p:02}/ndvi{p:02}_{dekads[i]}.tif'
            
            with rasterio.open(file,"w",**profile) as dst:
                dst.write(im,1)
    return


    

