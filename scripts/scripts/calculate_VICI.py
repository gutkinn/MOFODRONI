import rasterio
from pathlib import Path
import os
import glob
import numpy as np
import shutil
import geopandas as gpd
from rasterio.warp import reproject, calculate_default_transform, Resampling
from rasterio.mask import mask
from rasterio.merge import merge
from shapely.geometry import box
import geopandas as gpd
from rasterio.features import geometry_mask

from rasterio.enums import Resampling
import math

def get_landcover(bbox,geom,destination):
    
    landcover_folder = "/home/eoafrica/eodata/auxdata/ESA_WORLD_COVER/2021"
    
    # Round min values down to nearest multiple of 3
    rounded_minX = (np.floor(bbox[0] / 3) * 3).astype(int)
    rounded_minY = (np.floor(bbox[1] / 3) * 3).astype(int)

    # Round max values up to nearest multiple of 3
    rounded_maxX = (np.ceil(bbox[2] / 3) * 3).astype(int)
    rounded_maxY = (np.ceil(bbox[3] / 3) * 3).astype(int)
    
    E_range = np.arange(rounded_minX,rounded_maxX,3)
    N_range = np.arange(rounded_minY,rounded_maxY,3)
    
    for E in E_range:
        for N in N_range:
            tile = f"ESA_WorldCover_10m_2021_v200_N{N:02d}E{E:03d}"
            tile_file = os.path.join(landcover_folder,tile,f"{tile}_Map.tif")
            destination_path = os.path.join(destination,f"{tile}.tif")
            if not os.path.exists(destination_path):
                if os.path.exists(tile_file):
                    shutil.copy(tile_file, destination_path)

                with rasterio.open(destination_path) as src:
                    out_image, out_transform = rasterio.mask.mask(src, geom, crop=True)
                    out_meta = src.meta

                    out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

                with rasterio.open(destination_path, "w", **out_meta) as dest:
                    dest.write(out_image)     
                    
def get_landcover_EW(bbox,geom,destination):
    landcover_folder = "/home/eoafrica/eodata/auxdata/ESA_WORLD_COVER/2021"
    
    # Round min values down to nearest multiple of 3
    rounded_minX = (np.floor(bbox[0] / 3) * 3).astype(int)
    rounded_minY = (np.floor(bbox[1] / 3) * 3).astype(int)

    # Round max values up to nearest multiple of 3
    rounded_maxX = (np.ceil(bbox[2] / 3) * 3).astype(int)
    rounded_maxY = (np.ceil(bbox[3] / 3) * 3).astype(int)
    
    E_range = np.arange(rounded_minX,rounded_maxX,3)
    W_range = np.array([f'W{abs(val):03d}' for val in E_range if val < 0])
    E_range = np.array([f'E{abs(val):03d}' for val in E_range if val >= 0])
    EW_range = np.concatenate([E_range,W_range])
    N_range = np.arange(rounded_minY,rounded_maxY,3)
    for E in EW_range:
        for N in N_range:
            tile = f"ESA_WorldCover_10m_2021_v200_N{N:02d}{E}"
            tile_file = os.path.join(landcover_folder,tile,f"{tile}_Map.tif")
            destination_path = os.path.join(destination,f"{tile}.tif")
            if not os.path.exists(destination_path):
                if os.path.exists(tile_file):
                    shutil.copy(tile_file, destination_path)

                with rasterio.open(destination_path) as src:
                    out_image, out_transform = rasterio.mask.mask(src, geom, crop=True)
                    out_meta = src.meta

                    out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

                with rasterio.open(destination_path, "w", **out_meta) as dest:
                    dest.write(out_image)     

# Function to process each raster
def process_raster(input_raster,output_raster,template_dict):
    with rasterio.open(input_raster) as src:
        data = src.read()
        data[data!=40]=0
        data[data==40]=1
        
        new_width = math.ceil(src.width * src.res[0]/template_dict['resolution'][0])
        new_height = math.ceil(src.height * src.res[1]/template_dict['resolution'][1])
        
        # Reproject to match template
        transform, width, height = calculate_default_transform(
            src.crs, template_dict["crs"], new_width, new_height,*src.bounds
        )
        
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": template_dict["crs"],
            "transform": transform,
            "width": width,
            "height": height,
            "nodata": np.nan,
            "dtype": 'float32'
        })

        resampled_data = np.empty((height, width), dtype='float64')

        reproject(
            source=data,
            destination=resampled_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=template_dict["crs"],
            resampling=Resampling.min  # Apply averaging resampling
        )

        # Save output raster
        output_folder = Path(output_raster)
        output_path = os.path.join(output_folder, 'cropland',os.path.basename(input_raster))

        with rasterio.open(output_path, "w", **kwargs) as dst:
            dst.write(data[0], 1)

        print(f"Processed and saved: {output_path}")    
        
def merge_cropmasks(destination,template_dict):
    
    cropland_folder = os.path.join(destination,"cropland")
    merge_out = os.path.join(cropland_folder,'WorldCover_merged.tif')
    list_masks = glob.glob(os.path.join(cropland_folder,'*.tif'))
    list_masks_open = [rasterio.open(mask,'r') for mask in list_masks]

    merged, out_trans = merge(list_masks_open)

    out_meta = list_masks_open[0].meta.copy()
    
    transform, width, height = calculate_default_transform(
            out_meta['crs'], template_dict["crs"], template_dict["width"], template_dict["height"],*template_dict["extent"]
        )
    
    out_meta.update({
            "driver": "GTiff",
            "crs":template_dict["crs"],
            "height": height,
            "width": width,
            "transform": transform,
            "nodata": np.nan,
            "dtype": 'float32'
        })
    
    resampled_merged = np.empty((height, width), dtype='float64')

    reproject(
            source=merged,
            destination=resampled_merged,
            src_transform=out_trans,
            src_crs=list_masks_open[0].meta['crs'],
            dst_transform=transform,
            dst_crs=template_dict["crs"],
            resampling=Resampling.nearest  # Apply averaging resampling
        )
    
    with rasterio.open(merge_out, 'w', **out_meta) as dest:
        dest.write(resampled_merged,indexes=1)
        
    print(f'Cropland masks merged and saved: {merge_out}')
    
    
def get_cropland_mask(destination,template):
    
    cropland_folder = os.path.join(destination,"cropland")
    os.makedirs(cropland_folder,exist_ok=True)
    
    landcover_files = glob.glob(os.path.join(destination,"*.tif"))
    for landcover_file in landcover_files:

        print(landcover_file)
        cropland_file = os.path.join(cropland_folder, f"{os.path.basename(landcover_file)}")        
        process_raster(landcover_file,destination,template)


def vici(ndvi,lower,upper):

    vici_value = ((ndvi.astype('float32')-upper.astype('float32'))/(lower.astype('float32')-upper.astype('float32')))*100

    vici_value[vici_value == float('inf')] = 0
    vici_value[vici_value < 0] = 0
    vici_value[vici_value > 100] = 100

    return(vici_value)

def calculate_VICI(CPSZ_file,date,lower_threshold,upper_threshold,VICI_folder,output_folder,bbox,all_dates,onlyCropland = False, processMasks = False):

    with rasterio.open(CPSZ_file,'r') as src:
        cpsz = src.read(1)
    
    cluster_folder = Path(VICI_folder).parent

    ndvi_file = os.path.join(cluster_folder,"Adj_NDVI_stack.tif")

    lower_folder = os.path.join(cluster_folder,"percentiles",lower_threshold)
    upper_folder = os.path.join(cluster_folder,"percentiles",upper_threshold)
    
    year,month,day = date.split('-')
    
    dekad = f"{month}-{day}"

    cropland_mask = os.path.join(output_folder,"landcover","cropland","WorldCover_merged.tif")

    lower_file = os.path.join(lower_folder,f"ndvi{lower_threshold[-2:]}_{month}{day}.tif")
    upper_file = os.path.join(upper_folder,f"ndvi{upper_threshold[-2:]}_{month}{day}.tif")
    
    with rasterio.open(ndvi_file) as src:
        ndvi = src.read(all_dates.index(date))
        profile = src.profile

    with rasterio.open(lower_file) as src:
        lower = src.read(1)
        
    with rasterio.open(upper_file) as src:
        upper = src.read(1)

    vici_raster = vici(ndvi,lower,upper)
    vici_raster[np.isnan(lower)]=np.nan
    vici_raster[lower==255]=np.nan
    vici_raster[ndvi > 250]=np.nan # missing NDVI values
    vici_raster[lower==253]=np.nan # out of season
    
    """if onlyCropland:
        with rasterio.open(cropland_mask) as src:
            c_mask = src.read(1)
        vici_raster[c_mask==0]=254 # out of cropland"""
        
    #vici_raster[cpsz==0]= np.nan # out of CPSZ
    #vici_raster[c_mask==0]=np.nan
    vici_raster = vici_raster.astype(np.float32)
    
    vici_file = os.path.join(VICI_folder,f"VICI-{year}-{month}-{day}.tif")
    profile.pop("nodata", None)
    profile["nodata"] = np.nan
    profile["dtype"]=np.float32
    profile["count"]=1
    with rasterio.open(vici_file,"w",**profile) as dst:
        dst.write(vici_raster,1)

def calculate_SWIA(CPSZ_file,date,lower_threshold,upper_threshold,VICI_folder,output_folder,bbox,all_dates):

    with rasterio.open(CPSZ_file,'r') as src:
        cpsz = src.read(1)
    
    cluster_folder = Path(VICI_folder).parent

    ndvi_file = os.path.join(cluster_folder,"Adj_SWI_stack.tif")

    lower_folder = os.path.join(cluster_folder,"percentiles_swia",lower_threshold)
    upper_folder = os.path.join(cluster_folder,"percentiles_swia",upper_threshold)
    
    year,month,day = date.split('-')
    
    dekad = f"{month}-{day}"

    cropland_mask = os.path.join(output_folder,"landcover","cropland","WorldCover_merged.tif")

    lower_file = os.path.join(lower_folder,f"ndvi{lower_threshold[-2:]}_{month}{day}.tif")
    upper_file = os.path.join(upper_folder,f"ndvi{upper_threshold[-2:]}_{month}{day}.tif")
    
    with rasterio.open(ndvi_file) as src:
        ndvi = src.read(all_dates.index(date))
        profile = src.profile

    with rasterio.open(lower_file) as src:
        lower = src.read(1)
        
    with rasterio.open(upper_file) as src:
        upper = src.read(1)

    vici_raster = vici(ndvi,lower,upper)
    vici_raster[np.isnan(lower)]=np.nan
    vici_raster[lower==255]=np.nan
    vici_raster[ndvi > 250]=np.nan # missing NDVI values
    #vici_raster[lower==253]=np.nan # out of season

        
    #vici_raster[cpsz==0]= np.nan # out of CPSZ
    #vici_raster[c_mask==0]=np.nan
    vici_raster = vici_raster.astype(np.float32)
    
    vici_file = os.path.join(VICI_folder,f"SWIA-{year}-{month}-{day}.tif")
    profile.pop("nodata", None)
    profile["nodata"] = np.nan
    profile["dtype"]=np.float32
    profile["count"]=1
    with rasterio.open(vici_file,"w",**profile) as dst:
        dst.write(vici_raster,1)
        
def calculate_VICI_old(date,lower_threshold,upper_threshold,VICI_folder,output_folder,bbox,onlyCropland = True, processMasks = False):
    
    cluster_folder = Path(VICI_folder).parent
    cropped_country = os.path.join(output_folder,"invalid_pixel")
    season_folder = os.path.join(cluster_folder,"season_masks")
    lower_folder = os.path.join(cluster_folder,"cluster_stats",lower_threshold)
    upper_folder = os.path.join(cluster_folder,"cluster_stats",upper_threshold)
    
    year,month,day = date.split('-')
    
    dekad = f"{month}-{day}"
    
    ndvi_file = os.path.join(cropped_country,str(year),f"ndvi_upper_envelope_{year}-{month}-{day}_Mali.tif")
    cropland_mask = os.path.join(output_folder,"landcover","cropland","WorldCover_merged.tif")
    season_mask = os.path.join(season_folder,f"season-{month}-{day}.tif")
    lower_file = os.path.join(lower_folder,f"ndvi_{lower_threshold}-{month}-{day}.tif")
    upper_file = os.path.join(upper_folder,f"ndvi_{upper_threshold}-{month}-{day}.tif")
    
    with rasterio.open(ndvi_file) as src:
        ndvi = src.read()
        profile = src.profile
        
        template_dict = {
            "width":src.width,
            "height":src.height,
            "crs":src.crs,
            "transform":src.transform,
            "extent":src.bounds,
            "resolution":src.res,
            'dtype':profile["dtype"]
        }
        
    with rasterio.open(season_mask) as src:
        s_mask = src.read()

    with rasterio.open(cropland_mask) as src:
        c_mask = src.read()
    
    with rasterio.open(lower_file) as src:
        lower = src.read()
        
    with rasterio.open(upper_file) as src:
        upper = src.read()

    vici_raster = vici(ndvi,lower)
    vici_raster[np.isnan(s_mask)]=np.nan
    vici_raster[s_mask==0]=np.nan
    #vici_raster[c_mask==0]=np.nan
    vici_raster = vici_raster.astype(np.float32)
    
    vici_file = os.path.join(VICI_folder,f"VICI-{year}-{month}-{day}.tif")
    profile.pop("nodata", None)
    profile["nodata"] = np.nan
    profile["dtype"]=np.float32
    with rasterio.open(vici_file,"w",**profile) as dst:
        dst.write(vici_raster)


    
    
    
    