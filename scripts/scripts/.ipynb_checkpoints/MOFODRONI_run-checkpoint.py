#Importing the required packages
import glob
import os
from pathlib import Path
from osgeo import gdal
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import math
import importlib
import seaborn as sns
import shutil

import prepare_country_VICI
import calculate_VICI
import crop_raster
import stack_tif
import kmeans
import cluster_stats
import seasons
import adj_factors

print("All packages imported successfully.")

run_clipping = False
run_cropland = False
run_invalid_masking = False
run_clustering = False
run_stats = False
run_adjustments = False
run_percentiles = False
correct_seasons = False

def flatten(xss):
    return [x for xs in xss for x in xs]

def make_dirs(main_dir):
    output_dir = os.path.join(main_dir,"Output_new")
    os.makedirs(output_dir,exist_ok = True)
    savgol_folder = os.path.join(main_dir,"SavGol")
    cropped_folder = os.path.join(output_dir,"cropped_country")
    invalid_pixel_folder = os.path.join(output_dir,"invalid_pixel")
    os.makedirs(invalid_pixel_folder,exist_ok=True)
    analysis_folder = os.path.join(output_dir,"analysis")
    os.makedirs(analysis_folder, exist_ok = True)

    return output_dir, savgol_folder, cropped_folder, invalid_pixel_folder, analysis_folder
    
def create_zones(cluster_folder,output_dir,invalid_pixel_folder,num_zones,run_clustering,flag_harmonics):
    
    output_cube = os.path.join(cluster_folder,"cube_to_cluster.img")

    output_CPSZ = os.path.join(cluster_folder,f"CPSZ_{num_zones}zones.tif")

    #Create stack of rasters
    stack_folder = os.path.join(output_dir,"stacks")
    os.makedirs(stack_folder,exist_ok=True)
        
    cropped_files = sorted(glob.glob(os.path.join(cropped_folder,"*","*"+country+".tif")))
    invalid_pix_files = sorted(glob.glob(os.path.join(invalid_pixel_folder,"*","*"+country+".tif")))
    cropped_stack = os.path.join(stack_folder,"cropped_NDVI_stack.tif")
    invalid_stack = os.path.join(stack_folder,"invalid_pixel_stack.tif")
    
    if run_clustering:
        #scale flag used to scale NDVI from 0-250 to 0-1, not needed for invalid pixels 
        if not os.path.isfile(cropped_stack):
            stack_tif.merge_tifs_to_multiband(cropped_files,cropped_stack)
        if not os.path.isfile(invalid_stack):
            stack_tif.merge_tifs_to_multiband(invalid_pix_files,invalid_stack)
        
        #Create clusters - when rerunning, make sure that the cube is removed:
        #if os.path.isfile(output_cube):
        #    os.remove(output_cube)
            
        if not os.path.isfile(output_CPSZ):
            kmeans.cluster_KMeans(cropped_stack,invalid_stack,output_cube,output_CPSZ,num_zones,flag_harmonics)
        print("KMeans clustering finished")    
    
    return output_CPSZ, invalid_stack, stack_folder
    
def define_seasons(output_CPSZ,stack_folder,stats_folder,cluster_folder,num_zones,flag_harmonics):
    #Define Seasons
    #Create median stack
    median_stack = os.path.join(stack_folder,f"median_stack_harm{flag_harmonics}_{num_zones}zones.tif")
    median_files = sorted(glob.glob(os.path.join(stats_folder,"p50","*.tif")))
        
    stack_tif.merge_tifs_to_multiband(median_files,median_stack)
        
    with rasterio.open(output_CPSZ) as src:
        cps = src.read()
    zones = np.unique(cps)
        
    seasons_file = os.path.join(stats_folder,"seasons.csv")
        
    seasons.get_seasons(median_stack,output_CPSZ,seasons_file)
        
    seasons_folder = os.path.join(cluster_folder,"season_masks")
    os.makedirs(seasons_folder,exist_ok=True)
        
    seasons.season_mask(output_CPSZ,seasons_file,seasons_folder)
    
#Determine variables
country = "Nigeria"
year = 2024

main_dir = Path(os.getcwd()).parent
output_dir, savgol_folder, cropped_folder, invalid_pixel_folder, analysis_folder = make_dirs(main_dir)

nigeria_shp = os.path.join(main_dir, 'shapefiles','nigeria_country.shp')
nigeria_vector = gpd.read_file(nigeria_shp)['geometry']

#BoundingBox - Set values to the desired bbox
minX = 2.6 #2.6
maxX = 14.5 #14.5
minY = 4.2 #4.2
maxY = 13.8 #13.8
bbox = [minX,minY,maxX,maxY]
print(f'Location is {country}, year is {year}')

if run_clipping:
    #Cropping the NDVI data to the bbox
    error_list = []
    #Create for-loop here to also loop over years 1999-2024
    for year in range(1999,2025):

        print(f'Processing year {year}')
        savgol_y = os.path.join(savgol_folder,str(year))
        output_y = os.path.join(output_dir,"cropped",str(year))
        country_y = os.path.join(output_dir,"cropped_country",str(year))
        os.makedirs(country_y,exist_ok=True)
        os.makedirs(output_y,exist_ok=True)

        savgol_files = glob.glob(os.path.join(savgol_y,"*.tif"))

        for file in savgol_files:
            try:
                #print(f'Processing {file}')
                input_file = file
                output_file = os.path.join(output_y,Path(file).stem+".tif").replace(".tif",f"_{country}.tif")
                country_file = os.path.join(country_y,Path(file).stem+".tif").replace(".tif",f"_{country}.tif")
                crop_raster.crop(input_file,output_file,bbox)
                with rasterio.open(output_file) as src:
                    out_image, out_transform = rasterio.mask.mask(src, nigeria_vector, crop=True)
                    out_meta = src.meta
                out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
                with rasterio.open(country_file, "w", **out_meta) as dest:
                    dest.write(out_image)

            except Exception as e:
                print(e)
                error_list.append(file)
                continue
        print(f'Year {year} completed successfully.')
        

#Creating a masking file for invalid pixels.

if run_invalid_masking:

    cropped_files = sorted(glob.glob(os.path.join(cropped_folder,"*","*.tif")))
    for cropped_file in cropped_files:
        invalid_file = os.path.join(invalid_pixel_folder,Path(cropped_file).parent.stem,Path(cropped_file).stem+".tif")
        print(invalid_file)
        with rasterio.open(cropped_file) as src:
            data = src.read()
            profile = src.profile

        invalid = data.copy()
        invalid[np.isnan(invalid)]=255
        invalid = invalid.astype("uint8")

        os.makedirs(Path(invalid_file).parent,exist_ok=True)
        
        with rasterio.open(invalid_file, "w", **profile) as dst:
            dst.write(invalid[0,:,:],1)
            
            template_dict = {
            "width":dst.width,
            "height":dst.height,
            "crs":dst.crs,
            "transform":dst.transform,
            "extent":dst.bounds,
            "resolution":dst.res,
            'dtype':profile["dtype"]
            }
        #generate cropland masks
        if run_cropland:
            landcover_folder = os.path.join(output_dir,"landcover")
            os.makedirs(landcover_folder,exist_ok=True)
            
            calculate_VICI.get_landcover(bbox,landcover_folder)
            calculate_VICI.get_cropland_mask(landcover_folder,template_dict)
            calculate_VICI.merge_cropmasks(landcover_folder,template_dict)
            run_cropland = False
        
    print("Invalid Pixel Mask is created")
    
flag_harmonics = True

for u_thresh in range(15,55,5):
    upper_threshold = "p"+str(u_thresh)
    sub_an = "analysis_05_"+str(u_thresh)
    
    out_folder = os.path.join(output_dir,"analysis",sub_an)
            
    for num_zones in range(20,140,20):

        cluster_folder = os.path.join(out_folder,f"{num_zones:03}zones")
        os.makedirs(cluster_folder,exist_ok=True)

        print(f'Running for {num_zones} zones')
            
        output_CPSZ, invalid_stack, stack_folder = create_zones(cluster_folder,output_dir,invalid_pixel_folder,num_zones,run_clustering,flag_harmonics)

        C_Adj_file = os.path.join(stack_folder,'C_stack_adj.tif')
        Adjusted_NDVI_file = os.path.join(stack_folder,'Adj_NDVI_stack.tif')

        with rasterio.open(output_CPSZ,'r') as src:
            cpsz = src.read()
        zones=np.unique(cpsz)[1:]
        print(f'Total {len(zones)} zones.')

        stats_folder = os.path.join(cluster_folder,"cluster_stats")
        os.makedirs(stats_folder,exist_ok=True)
        stats_list = ["lta","std","p05","p10","p15","p20","p25","p50","p75","p90"]
        
        if run_stats:
            print('Running stats.')
            cluster_stats.cluster_stats(output_CPSZ,invalid_stack,stats_folder,stats_list)

        if run_adjustments:
            print('Running compute adjustments')
            adj_factors.compute_adjustments(invalid_stack,output_CPSZ,C_Adj_file,cluster_folder)

            print('Aplying correction factor')
            adj_factors.apply_correction_factor(invalid_stack, C_Adj_file, Adjusted_NDVI_file)
            
        outdir_percentiles = os.path.join(cluster_folder,"percentiles")
        os.makedirs(os.path.join(cluster_folder,"percentiles"),exist_ok=True)
        percentile_numbers = ['05', str(u_thresh)]

        if run_percentiles:
            print('Running percentiles')
            percentiles, p50_array = adj_factors.compute_percentiles(Adjusted_NDVI_file, output_CPSZ, percentile_numbers, outdir_percentiles)

        if correct_seasons:
            seasons_file = os.path.join(stats_folder,"seasons.csv")
            seasons_arr = adj_factors.get_seasons(p50_array, zones, seasons_file)
            adj_factors.correct_percentiles_for_seasons(output_CPSZ, seasons_arr,percentiles, percentile_numbers, outdir_percentiles)

        all_dates = [val.split('_')[-2] for val in sorted(glob.glob(os.path.join(invalid_pixel_folder,'*','*.tif')))]
        VICI_folder = os.path.join(cluster_folder,"VICI")
        print(f'out vici folder: {VICI_folder}')
        os.makedirs(VICI_folder,exist_ok=True)
        dekads = cluster_stats.get_dekads()
        years = [2020,2021,2022,2023]#[y for y in range(1999,2025)]
        
        lower_threshold,upper_threshold = [f"p{n:02}" for n in percentile_numbers]
        onlyCropland = False
        processMasks=True
        
        for year in years:
            print(f'Calculating VICI for {year}')
            for dekad in dekads:
                
                date = f'{year}{dekad}'
        
                calculate_VICI.calculate_VICI(output_CPSZ,date,lower_threshold,upper_threshold,VICI_folder,output_dir,bbox,all_dates,onlyCropland,processMasks)
        
        shutil.rmtree(stats_folder)
    
    