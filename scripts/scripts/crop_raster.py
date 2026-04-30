from osgeo import gdal
import sys

def crop(input_raster, output_raster, bbox):
    """
    Crop a raster file to a specified bounding box.
    
    :param input_raster: Path to the input raster file
    :param output_raster: Path to save the cropped raster file
    :param bbox: Bounding box (minX, minY, maxX, maxY)
    """
    minX, minY, maxX, maxY = bbox
    
    gdal.Translate(
        output_raster,
        input_raster,
        projWin=[minX, maxY, maxX, minY]
    )

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python crop_raster.py <input_raster> <output_raster> <minX> <minY> <maxX> <maxY>")
        sys.exit(1)
    
    input_raster = sys.argv[1]
    output_raster = sys.argv[2]
    bbox = tuple(map(float, sys.argv[3:]))
    
    crop(input_raster, output_raster, bbox)
