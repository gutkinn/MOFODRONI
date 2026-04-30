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
import xarray as xr
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
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

def flatten(xss):
    return [x for xs in xss for x in xs]

def create_drought_limits(year,drought_months):
    drought_limits = {'start_drought': [f'{year}-{drought_months[0]}-01'],
                      'end_drought': [f'{year}-{f'{int(drought_months[-1])+1:02d}'}-01']}
    drought_limits = pd.DataFrame(drought_limits)
    for col in drought_limits.columns:
        drought_limits[col] = pd.to_datetime(drought_limits[col])
    return drought_limits

def rearrange_plotvals(in_plots,year):
    plots = in_plots.copy()
    plots['Drought degree'] = plots['C9']
    plots.drop(plots.loc[plots[f'?INC{year}'] == 'NIL'].index, inplace=True)
    plots[f'change{year}'] = plots[f'?INC{year}'].astype(float)

    plots[f'bin{year}'],bins = pd.qcut(plots[f'change{year}'],q=5,retbins=True,labels=False)
    bin_labels = [str(np.round(bins[0],1)) + ' to ' + str(np.round(bins[1],1)),
        str(np.round(bins[1],1)) + ' to ' + str(np.round(bins[2],1)),
        str(np.round(bins[2],1)) + ' to ' + str(np.round(bins[3],1)),
        str(np.round(bins[3],1)) + ' to ' + str(np.round(bins[4],1)),
        str(np.round(bins[4],1)) + ' to ' + str(np.round(bins[5],1))]

    plots[f'bin{year}'],bins = pd.qcut(plots[f'change{year}'],q=5,retbins=True,labels=bin_labels)
    
    return plots

def make_dirs(main_dir,foldername):
    output_dir = os.path.join(main_dir,foldername)
    os.makedirs(output_dir,exist_ok = True)
    savgol_folder = os.path.join(main_dir,"SM")
    cropped_folder = os.path.join(output_dir,"cropped_country_swia")
    os.makedirs(cropped_folder,exist_ok = True)
    invalid_pixel_folder = os.path.join(output_dir,"invalid_pixel_swia")
    os.makedirs(invalid_pixel_folder,exist_ok=True)
    analysis_folder = os.path.join(output_dir,"analysis")
    os.makedirs(analysis_folder, exist_ok = True)
    images_folder = os.path.join(output_dir,"images")
    os.makedirs(images_folder, exist_ok = True)

    return output_dir, savgol_folder, cropped_folder, invalid_pixel_folder, analysis_folder, images_folder


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

run_clustering = True
run_stats = True

main_dir = Path(os.getcwd()).parent
output_dir, savgol_folder, cropped_folder, invalid_pixel_folder, analysis_folder, out_images = make_dirs(main_dir,'Output')

nigeria_shp = os.path.join(main_dir, 'shapefiles','nigeria_country.shp')
nigeria_vector = gpd.read_file(nigeria_shp)['geometry']

#BoundingBox - Set values to the desired bbox
minX = 2.6 #2.6
maxX = 14.5 #14.5
minY = 4.2 #4.2
maxY = 13.8 #13.8
bbox = [minX,minY,maxX,maxY]

print(output_dir)

invalid_stack = r'/home/eoafrica/shared/MOFODRONI/Output/stacks/invalid_pixel_stack_swia.tif'
onlyCropland = False
processMasks=True

for num_zones in [110]:
    out_folder = os.path.join(output_dir,"analysis_swia",f"{num_zones:02}zones")
    output_CPSZ = os.path.join(out_folder,f"CPSZ_{num_zones}zones.tif")
    print(f'Running for {num_zones} zones')
    
    for u_thresh in range(30,55,5):
        upper_threshold = "p"+str(u_thresh)
        sub_an = "analysis_05_"+str(u_thresh)
        cluster_folder = os.path.join(out_folder,sub_an)
        os.makedirs(cluster_folder,exist_ok=True)
        print(f'Running for {upper_threshold}')
                        
        C_Adj_file = os.path.join(cluster_folder,'C_stack_adj_swia.tif')
        Adjusted_NDVI_file = os.path.join(cluster_folder,'Adj_SWI_stack.tif')
            
        with rasterio.open(output_CPSZ,'r') as src:
            cpsz = src.read()
        zones=np.unique(cpsz)
        print(f'Total {len(zones)} zones.')
            
        print('Running stats.')
        stats_folder = os.path.join(cluster_folder,"cluster_stats_swia")
        os.makedirs(stats_folder,exist_ok=True)
                    
        stats_list = ["lta","std","p05","p10","p15","p20","p25","p50","p75","p90"]
        cluster_stats.cluster_stats(output_CPSZ,invalid_stack,stats_folder,stats_list)
            
        print('Running compute adjustments')
        adj_factors.compute_adjustments(invalid_stack,output_CPSZ,C_Adj_file,cluster_folder)
            
        print('Running compute adjustments')
        adj_factors.apply_correction_factor(invalid_stack, C_Adj_file, Adjusted_NDVI_file)
            
        print('Running percentiles')
        outdir_percentiles = os.path.join(cluster_folder,"percentiles_swia")
        os.makedirs(outdir_percentiles,exist_ok=True)
        percentile_numbers = ['05', str(u_thresh)]
        percentiles, p50_array = adj_factors.compute_percentiles(Adjusted_NDVI_file, output_CPSZ, percentile_numbers, outdir_percentiles)
            
        seasons_file = os.path.join(stats_folder,"seasons.csv")
        seasons_arr = adj_factors.get_seasons(p50_array, zones, seasons_file)
        adj_factors.correct_percentiles_for_seasons(output_CPSZ, seasons_arr,percentiles, percentile_numbers, outdir_percentiles)
            
        all_dates = [val.split('_')[-2] for val in sorted(glob.glob(os.path.join(invalid_pixel_folder,'*','*.tif')))]
        VICI_folder = os.path.join(cluster_folder,"SWIA")
        print(f'out vici folder: {VICI_folder}')
        os.makedirs(VICI_folder,exist_ok=True)
        dekads = cluster_stats.get_dekads()
        years = [y for y in range(2019,2025)]
                    
        lower_threshold,upper_threshold = [f"p{n:02}" for n in percentile_numbers]
        
        for year in years:
            for dekad in dekads:
                date = f'{year}{dekad}'
                calculate_VICI.calculate_SWIA(output_CPSZ,date,lower_threshold,upper_threshold,VICI_folder,output_dir,bbox,all_dates)
                
        os.remove(Adjusted_NDVI_file)
        os.remove(C_Adj_file)
        shutil.rmtree(os.path.join(cluster_folder,'adjustments'))
        shutil.rmtree(os.path.join(cluster_folder,'percentiles_swia'))
        shutil.rmtree(os.path.join(cluster_folder,'cluster_stats_swia'))

    
