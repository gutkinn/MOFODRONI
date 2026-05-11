import os
import rasterio
from rasterio.merge import merge
import glob
import numpy as np

def merge_tifs_to_multiband(tif_files, output_file, scale=False):
    """
    Finds all .tif files in a folder and stores them as separate bands in a multi-band TIFF file.

    Parameters:
    - input_folder: The folder containing the .tif files.
    - output_file: The path to save the multi-band TIFF file.
    """
    # Read each TIFF file as a separate band
    datasets = [rasterio.open(tif) for tif in tif_files]
    band_names = [os.path.basename(tif).split('.')[0] for tif in tif_files]  # Extract filenames without extensions

    
    with rasterio.open(tif_files[0]) as src:
        profile = src.profile
        data = src.read()
        width = src.width
        height = src.height

    for ds in datasets:
        if ds.width != width or ds.height != height:
            print(f"Error: {ds.name} has different dimensions. All files must have the same size.")
            return

    # Create an empty array to hold all bands
    stacked_data = np.array([ds.read(1) for ds in datasets])

    # Update the profile for a multi-band TIFF
    profile.update(count=len(datasets), dtype=stacked_data.dtype)

    # Write the multi-band TIFF
    with rasterio.open(output_file, "w", **profile) as dst:
        for i, band in enumerate(stacked_data, start=1):
            dst.write(band, i)
            
            # Store band names in metadata
            dst.update_tags(**{f'Band_{i+1}': name for i, name in enumerate(band_names)})

    print(f"Multi-band TIFF saved as: {output_file}")

    # Close datasets
    for ds in datasets:
        ds.close()
