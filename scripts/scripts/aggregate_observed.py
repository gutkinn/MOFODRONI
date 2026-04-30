import os
import glob
import rasterio
import warnings

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from rasterstats import zonal_stats
from datetime import datetime

def run_execute(run_dict):

    warnings.filterwarnings("ignore")

    overwrite = run_dict["overwrite_aggregate_obs"]

    country = run_dict["country"]
    if country == "Somalia":
        country_abb = "SOM"
    elif country == "Mali":
        country_abb = "MLI"
    elif country == "Mozambique":
        country_abb = "MOZ"
    
    resamp_folder = os.path.join("/vitodata/CENTAUR/resampling",country,"VICI_V3")
    agg_folder = "/vitodata/CENTAUR/aggregation/"

    VICI_files = glob.glob(os.path.join(resamp_folder,"*.tif"))
    VICI_files_coldCase = [file for file in VICI_files if datetime.strptime(file.split("_")[-1].split(".")[0],"%Y-%m-%d")>=datetime.strptime(str(run_dict["splitYear"]),"%Y")]
    VICI_files_coldCase = VICI_files

    #WorldCover is used to determine the number of pixels in each spatial unit
    worldcover = "/vitodata/CENTAUR/resampling"
    worldcover = os.path.join(worldcover,run_dict["country"],"worldcover","worldcover_resampled.tif")

    with rasterio.open(worldcover) as src:
        worldcover = src.read(1)

    cropland_mask = (worldcover == 40)
    grassland_mask = (worldcover == 30)

    #Model determines different values for cropland/grassland
    lc_indicators = []
    if run_dict["indicator_dict"]["cropland"]:
        lc_indicators.append("cropland")
    if run_dict["indicator_dict"]["grassland"]:
        lc_indicators.append("grassland")

        for lc_indicator in lc_indicators:
            if lc_indicator == "cropland":
                mask = cropland_mask
            elif lc_indicator == "grassland":
                mask = grassland_mask

            for adm_level in run_dict["adm_level"]: #ADM1/ADM2/ADM3

                administrative_regions = glob.glob(os.path.join(agg_folder,"Administrative Boundaries",run_dict["country"],f"*{adm_level}.shp"))[0]
                ADM_shp = gpd.read_file(administrative_regions)
                ADM1_shp = gpd.read_file(administrative_regions.replace(adm_level,"ADM1"))

                for file in tqdm(VICI_files_coldCase,desc=f"Aggregating Drought Indicator for {adm_level}"):

                    output_folder = os.path.join(agg_folder,"visuals","V3","ADR",run_dict["model_horizon"],run_dict["country"],adm_level,lc_indicator)
                    os.makedirs(output_folder,exist_ok=True)

                    output_file = os.path.join(output_folder,Path(file).stem+'.png')
                    st_ADR = Path(file).stem.replace("VICI","WFS_006_ADR_OBS")
                    drought_risk_folder = os.path.join(agg_folder,"coldCase","drought_risk",run_dict["model_horizon"])

                    if not os.path.exists(os.path.join(drought_risk_folder,country_abb+"_"+st_ADR+"_"+adm_level+"_"+lc_indicator+".geojson")) or overwrite:

                        with rasterio.open(file) as src:
                            transform = src.transform
                            affine = rasterio.Affine(transform[0],transform[1],transform[2],transform[3],transform[4],transform[5])
                            array = np.array(src.read()[0,:,:])

                        array = np.where(array==255,np.nan,array)
                        array = np.where(mask,array,np.nan)

                        num_pixels = pd.DataFrame(zonal_stats(administrative_regions,worldcover,affine=affine,stats="count"))

                        drought = np.where(array>0,1,array)
                        severe_drought = np.where(array==100,1,np.where(array<100,0,array))
                        drought_stats = pd.DataFrame(zonal_stats(administrative_regions,drought,affine=affine,stats="count sum"))
                        drought_stats["drought_risk"]=(drought_stats["sum"]/(drought_stats["count"]+1)).astype(float)
                        #drought_stats["drought_risk"]=np.where(np.isnan(drought_stats["drought_risk"]),0,drought_stats["drought_risk"])
                        sev_drought_stats = pd.DataFrame(zonal_stats(administrative_regions,severe_drought,affine=affine,stats="count sum"))
                        sev_drought_stats["sev_drought_risk"]=(sev_drought_stats["sum"]/(sev_drought_stats["count"]+1)).astype(float)
                        #sev_drought_stats["sev_drought_risk"]=np.where(np.isnan(sev_drought_stats["sev_drought_risk"]),0,sev_drought_stats["sev_drought_risk"])

                        #Filter for areas where cropland/grassland is less than 1%
                        drought_stats['num_pix']=num_pixels['count']
                        sev_drought_stats['num_pix']=num_pixels['count']

                        drought_stats['drought_risk'] = np.where(drought_stats['count']<0.01*drought_stats['num_pix'],np.nan,drought_stats['drought_risk'])
                        sev_drought_stats['sev_drought_risk'] = np.where(sev_drought_stats['count']<0.01*sev_drought_stats['num_pix'],np.nan,sev_drought_stats['sev_drought_risk'])

                        drought_stats["drought_risk"] = np.where(drought_stats["count"]==0,np.nan,drought_stats["drought_risk"])
                        sev_drought_stats["sev_drought_risk"] = np.where(sev_drought_stats["count"]==0,np.nan,sev_drought_stats["sev_drought_risk"])
                            
                        ADM_file = ADM_shp.copy()
                        ADM_file["drought_risk"]=drought_stats["drought_risk"]
                        ADM_file["severe_drought_risk"]=sev_drought_stats["sev_drought_risk"]
                        ADM_file["npixels"]=drought_stats["count"]
                        ADM_file = gpd.GeoDataFrame(ADM_file,crs=ADM_shp.crs)
                        #ADM_file["drought_risk"]=np.where(ADM_file["drought_risk"]==np.nan,0,ADM_file["drought_risk"])

                        drought_risk_folder = os.path.join(agg_folder,"coldCase","drought_risk",run_dict["model_horizon"])
                        severe_drought_risk_folder = os.path.join(agg_folder,"coldCase","severe_drought_risk",run_dict["model_horizon"])
                        
                        os.makedirs(drought_risk_folder,exist_ok=True)
                        os.makedirs(severe_drought_risk_folder,exist_ok=True)

                        '''

                        ax = ADM_file.plot(color="lightgray",edgecolor="lightgray")
                        ADM_file.plot(ax=ax,column="drought_risk",legend=True,cmap="RdYlGn_r",vmin=0,vmax=1)
                        ADM1_shp.plot(ax=ax,color="none",edgecolor="black",linewidth=0.5)
                        os.makedirs(Path(output_file).parent,exist_ok=True)
                        plt.title(Path(file).stem.split("_")[0])
                        # Add title to the colorbar
                        fig = ax.get_figure()
                        cax = fig.get_axes()[1]
                        cax.set_ylabel("ADR")
                        plt.savefig(output_file)
                        plt.close()

                        '''

                        st_ADR = Path(file).stem.replace("VICI","WFS_006_ADR_OBS")
                        st_SADR = st_ADR.replace("ADR","SADR")

                        output_shp = os.path.join(drought_risk_folder,country_abb+"_"+st_ADR+"_"+adm_level+"_"+lc_indicator+".shp")
                        #ADM_file.drop(["severe_drought_risk"],axis=1).to_file(output_shp)
                        ADM_file.drop(["severe_drought_risk"],axis=1).to_file(output_shp.replace(".shp",".geojson"),driver="GeoJSON")

                        '''        
                        ax = ADM_file.plot(color="lightgray",edgecolor="lightgray")
                        ADM_file.plot(ax=ax,column="severe_drought_risk",legend=True,cmap="RdYlGn_r",vmin=0,vmax=1)
                        ADM1_shp.plot(ax=ax,color="none",edgecolor="black",linewidth=0.5)
                        output_file = output_file.replace("ADR","SADR")
                        os.makedirs(Path(output_file).parent,exist_ok=True)
                        plt.title(Path(file).stem.split("_")[0])
                        # Add title to the colorbar
                        fig = ax.get_figure()
                        cax = fig.get_axes()[1]
                        cax.set_ylabel("SADR")
                        plt.savefig(output_file)
                        plt.close()
                        '''

                        output_shp = os.path.join(severe_drought_risk_folder,country_abb+"_"+st_SADR+"_"+adm_level+"_"+lc_indicator+".shp")
                        #ADM_file.drop(["drought_risk"],axis=1).to_file(output_shp)
                        ADM_file.drop(["drought_risk"],axis=1).to_file(output_shp.replace(".shp",".geojson"),driver="GeoJSON")