import numpy as np
import rasterio
import csv
import pandas as pd
import os
from tqdm import tqdm

def moving_average(x):
    w = np.ones(9)
    return np.convolve(x, w, 'valid') / 9

def get_seasons_2(p50_array, zones, seasons_file):

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

    return(seasons)
        
def season_mask_2(CPSZ,seasons,seasons_folder,percentiles,percentile_numbers):
    
    with rasterio.open(CPSZ) as src:
        cps = src.read().astype(int)
        profile = src.profile
    print(profile)
    zones = np.unique(cps)
    nzones = len(zones)
    
    days = ["01","11","21"]
    months = ["01","02","03","04","05",'06',"07","08","09","10","11","12"]
    dekads = []
    for month in months:
        for day in days:
            dekad = f"{month}-{day}"
            dekads.append(dekad)
    
    for i in range(len(dekads)):
        print('Dekad ' + str(i), end='\r')
        # Set percentile arrays to 255 (nodatavalue) when the zone is outside season
        for j in range(nzones):
            mask = cps == zones[j]
            season_flag = seasons[i, j]
            if season_flag == 0:
                for p in percentile_numbers:
                    percentiles[f'p{p:02}'][i, mask == True] = 255

        # Output percentiles files
        for p in percentile_numbers:
            im = percentiles[f'p{p:02}'][i, :, :]

            dekad_file = os.path.join(seasons_folder,f"season-{dekad}.tif")
            with rasterio.open(dekad_file,"w",**profile) as dst:
                dst.write(im)

    return

def get_seasons(median_stack,cps,seasons_file,isfile=False):
    
    if isfile:
        with rasterio.open(median_stack) as src:
            median = src.read()
    else:
        median = median_stack
        
    NDVI_thr = 70
    ratio_thr = 0.95
    
    zones = np.unique(cps)
    dekads = median.shape[0]
    
    seasons = np.zeros((dekads, len(zones)))
    
    for i,zone in enumerate(zones):
        zone_mask = cps[0,:,:] == zone
        zone_median = median[:,zone_mask]
        zone_profile = []
        for dekad in range(dekads):
            dekad_zone_value = np.nanmedian(zone_median[dekad,:])
            zone_profile.append(dekad_zone_value)

        max_val = np.max(zone_profile)
        zone_profile2 = np.tile(zone_profile, 2)  # Repeat data
        ma9 = moving_average(zone_profile2)

        zone_profile = zone_profile2[36:] # ignore first year of data
        ma9 = ma9[28:] # window of 9 
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
        
    return(seasons)
        
def season_mask(CPSZ,seasons_file,seasons_folder):
    
    with rasterio.open(CPSZ) as src:
        cps = src.read().astype(int)
        profile = src.profile
    zones = np.unique(cps)
        
    days = ["01","11","21"]
    months = ["01","02","03","04","05",'06',"07","08","09","10","11","12"]
    dekads = []
    for month in months:
        for day in days:
            dekad = f"{month}-{day}"
            dekads.append(dekad)
    
    data = pd.read_csv(seasons_file)
    data = data.rename(columns=lambda x: 'z' + str(x))
    print(data.columns)
    for i,dekad in tqdm(enumerate(dekads),desc="Creating Season Masks"):
        dekad_raster = np.empty(cps.shape)
        for zone in zones:
            zone_dekad = data[f"z{str(zone)}"][i]
            zone_mask = cps[0,:,:] == zone
            dekad_raster[0,zone_mask] = zone_dekad
        
        dekad_file = os.path.join(seasons_folder,f"season-{dekad}.tif")
        with rasterio.open(dekad_file,"w",**profile) as dst:
            dst.write(dekad_raster)
    
    
        
    
    
    