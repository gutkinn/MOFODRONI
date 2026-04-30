import numpy as np
from sklearn.cluster import MiniBatchKMeans
from osgeo import gdal
import pandas as pd
from prepare_country_VICI import get_statistics_cube
from prepare_country_VICI import get_harmonic_cube
import os
import rasterio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def cluster_KMeans(filtered_cropped_file, invalid_pixels_mask, cube_to_cluster_file, output_file, n_clusters, flag_harmonics, batch_size=10000):
    print(flag_harmonics)
    # Create dekadal stacks containing 20 years of observations per pixel
    if not os.path.isfile(cube_to_cluster_file):
        print("Creating Statistics Cube first")
        get_statistics_cube(filtered_cropped_file, invalid_pixels_mask, cube_to_cluster_file)
        
    if flag_harmonics:
        
        harmonics_cube_path = invalid_pixels_mask.replace("invalid_pixel_stack.tif","harmonics_cube.tif")
        if not os.path.isfile(harmonics_cube_path):
            print("Creating Harmonics Cube")
            get_harmonic_cube(invalid_pixels_mask,harmonics_cube_path)
        harmonics_cube = rasterio.open(harmonics_cube_path).read()
        harmonics_cube[np.where(harmonics_cube==0)]=np.nan
        for band in range(harmonics_cube.shape[0]):
            harmonics_cube[band,:,:] = np.interp(harmonics_cube[band], (np.nanmin(harmonics_cube[band]),
                                                                        np.nanmax(harmonics_cube[band])), (-1, +1))
        harmonics_cube = np.nan_to_num(harmonics_cube, nan=-9999)
    
        harmonics_transposed = np.transpose(harmonics_cube,(1,2,0))     
    
    dataset = gdal.Open(cube_to_cluster_file)
    
    # Extract band names from metadata
    metadata = dataset.GetMetadata()
    band_names = [metadata.get(f'Band_{i+1}', f'Timestep_{i+1}') for i in range(dataset.RasterCount)]
    if flag_harmonics:
        band_names+=['a0','amplitude1','phase1','amplitude2','phase2','amplitude3','phase3']

    # Read the bands into a numpy array
    bands = []
    for i in range(dataset.RasterCount):
        band = dataset.GetRasterBand(i + 1)
        bands.append(band.ReadAsArray())

    # Stack and reshape bands into a 2D array
    bands_2d = np.dstack(bands)
    print(f'bands_2 shape: {bands_2d.shape}')
    if flag_harmonics:
        bands_combined = np.concatenate((bands_2d, harmonics_transposed), axis=2) 
    else:
        bands_combined = bands_2d
        
    data = bands_combined.reshape(-1, bands_combined.shape[2])
    
    print("Performing KMeans Clustering")
    #kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, max_iter=1000, max_no_improvement=100, reassignment_ratio=0) VICI
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, max_iter=1000)
    kmeans.fit(data)

    # Get the cluster labels

    labels = kmeans.labels_
    zone_labels = np.unique(labels)

    remap_zones = {zone:list(zone_labels).index(zone) for zone in zone_labels}

    for key in remap_zones.keys():
        labels[labels==key] = remap_zones[key]

    # Reshape labels back to the original image shape
    clustered_image = labels.reshape(bands_2d.shape[:2])

    # Save the clustered image as a .tif file
    driver = gdal.GetDriverByName('GTiff')
    output_dataset = driver.Create(output_file, bands_2d.shape[1], bands_2d.shape[0], 1, gdal.GDT_Int32)

    # Copy geotransform, projection, and metadata from the input dataset
    output_dataset.SetGeoTransform(dataset.GetGeoTransform())
    output_dataset.SetProjection(dataset.GetProjection())
    output_dataset.SetMetadata(dataset.GetMetadata())

    output_dataset.GetRasterBand(1).WriteArray(clustered_image)
    output_dataset.FlushCache()
    output_dataset = None
    dataset = None
    
    centers_csv = output_file.replace(".tif",".csv")
    
    # Save cluster centers as CSV
    cluster_centers = kmeans.cluster_centers_
    df_centers = pd.DataFrame(cluster_centers, columns=band_names)
    df_centers.index.name = "Cluster"
    df_centers.to_csv(centers_csv)

    print(f"Cluster centers saved to {centers_csv}")