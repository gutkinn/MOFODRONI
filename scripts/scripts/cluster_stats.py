import rasterio
import numpy as np
import os

def load_raster(input_raster):
    with rasterio.open(input_raster) as src:
        data = src.read()
        profile = src.profile
        metadata = src.tags()  # Get metadata dictionary
        all_band_names = [metadata.get(f'Band_{i+1}', f'Band_{i+1}') for i in range(src.count)]

        return data, profile, all_band_names

def write_stats(stats_folder,stat_name,mean_raster,dekad,profile):
    lta_folder = os.path.join(stats_folder,stat_name)
    os.makedirs(lta_folder,exist_ok=True)
    mean_raster = mean_raster.reshape(1, mean_raster.shape[0], mean_raster.shape[1])
    average_raster = os.path.join(lta_folder,f"ndvi_{stat_name}{dekad}.tif")
    with rasterio.open(average_raster,"w",**profile) as dst:
        dst.write(mean_raster)

def get_dekads():
    days = ["01","11","21"]
    months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
    dekads = []
    for month in months:
        for day in days:
            dekad = f"-{month}-{day}"
            dekads.append(dekad)  
    return dekads
        
def cluster_stats(CPSZ_raster,invalid_pixel_mask,stats_folder,stats_list,wc=False):
    
    CPSZ, CPSZ_profile, CPSZ_bands = load_raster(CPSZ_raster)
    invalid, invalid_profile, invalid_bands = load_raster(invalid_pixel_mask)
    
    dekads = get_dekads()
    print(f'No. dekads: {len(dekads)}')
    
    unique_zones = np.unique(CPSZ)
    print(f'No. unique zones: {len(unique_zones)}')
    
    for dekad in dekads:
        
        dekad_indices = [i for i, name in enumerate(invalid_bands) if dekad in name]
        invalid_dekad = np.array([invalid[i,:,:] for i in dekad_indices])

        mean_raster = np.empty((CPSZ.shape[1],CPSZ.shape[2]))
        std_raster = mean_raster.copy()
        p05_raster = mean_raster.copy()
        p10_raster = mean_raster.copy()
        p15_raster = mean_raster.copy()
        p20_raster = mean_raster.copy()
        p25_raster = mean_raster.copy()
        p50_raster = mean_raster.copy()
        p75_raster = mean_raster.copy()
        p90_raster = mean_raster.copy()
        wc_raster = mean_raster.copy()
        
        for zone in unique_zones:
            
            zone_mask = CPSZ[0,:,:] == zone

            masked_data = invalid_dekad[:, zone_mask]
            masked_data = np.ravel(masked_data)
            masked_data = np.where(masked_data == 0,np.nan,masked_data)
            
            if "lta" in stats_list:
                zone_average = np.nanmean(masked_data)
                mean_raster[zone_mask] = zone_average
            if "std" in stats_list:
                zone_std = np.nanstd(masked_data)
                std_raster[zone_mask] = zone_std
            if 'p05' in stats_list:
                zone_p05 = np.nanpercentile(masked_data,5)
                p05_raster[zone_mask] = zone_p05
            if 'p10' in stats_list:
                zone_p10 = np.nanpercentile(masked_data,10)
                p10_raster[zone_mask] = zone_p10
            if 'p15' in stats_list:
                zone_p15 = np.nanpercentile(masked_data,15)
                p15_raster[zone_mask] = zone_p15
            if 'p20' in stats_list:
                zone_p20 = np.nanpercentile(masked_data,20)
                p20_raster[zone_mask] = zone_p20
            if 'p25' in stats_list:
                zone_p25 = np.nanpercentile(masked_data,25)
                p25_raster[zone_mask] = zone_p25
            if 'p50' in stats_list:
                zone_p50 = np.nanpercentile(masked_data,50)
                p50_raster[zone_mask] = zone_p50
            if 'p75' in stats_list:
                zone_p75 = np.nanpercentile(masked_data,75)
                p75_raster[zone_mask] = zone_p75
            if 'p90' in stats_list:
                zone_p90 = np.nanpercentile(masked_data,90)
                p90_raster[zone_mask] = zone_p90
            if wc:
                wc_val = stats_list[-1]
                wc_num = int(wc_val[1:])
                zone_wc = np.nanpercentile(masked_data,wc_num)
                wc_raster[zone_mask] = zone_wc
                
        if "lta" in stats_list:
            write_stats(stats_folder,"lta",mean_raster,dekad,CPSZ_profile)
        if "std" in stats_list:
            write_stats(stats_folder,"std",std_raster,dekad,CPSZ_profile)
        if "p05" in stats_list:
            write_stats(stats_folder,"p05",p05_raster,dekad,CPSZ_profile)
        if "p10" in stats_list:
            write_stats(stats_folder,"p10",p10_raster,dekad,CPSZ_profile)
        if "p15" in stats_list:
            write_stats(stats_folder,"p15",p15_raster,dekad,CPSZ_profile)
        if "p20" in stats_list:
            write_stats(stats_folder,"p20",p20_raster,dekad,CPSZ_profile)
        if "p25" in stats_list:
            write_stats(stats_folder,"p25",p25_raster,dekad,CPSZ_profile)
        if "p50" in stats_list:
            write_stats(stats_folder,"p50",p50_raster,dekad,CPSZ_profile)
        if 'p75' in stats_list:
            write_stats(stats_folder,"p75",p75_raster,dekad,CPSZ_profile)
        if 'p90' in stats_list:
            write_stats(stats_folder,"p90",p90_raster,dekad,CPSZ_profile)
        if wc:
            write_stats(stats_folder,wc_val,wc_raster,dekad,CPSZ_profile)
            
    print(f"Finished processing all dekads.")
            
            
            
