import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window
import numpy as np
import math
from rasterio.coords import BoundingBox
from rasterio.windows import from_bounds
from shapely.geometry import Polygon, mapping, shape

# Load the mining site polygons shapefile
mining_sites = gpd.read_file('/path')
print(mining_sites.crs)
mining_sites = mining_sites.to_crs('EPSG:4326')
print(mining_sites.crs)
# Directory containing the satellite images
image_dir = '/path'

# Function to ensure image dimensions are divisible by 32
def adjust_dimensions(image):
    rows, cols = image.shape
    new_rows = math.ceil(rows / 64) * 64
    new_cols = math.ceil(cols / 64) * 64
    adjusted_image = np.zeros((new_rows, new_cols), dtype=image.dtype)
    adjusted_image[:rows, :cols] = image
    return adjusted_image

# Approach - 1 (Extracting images and masks whose dimensions are divisible by 32)
# Loop over each polygon in the shapefile
for index, polygon in mining_sites.iterrows():
    # Convert the polygon geometry to GeoJSON format
    geo_json = [polygon['geometry'].__geo_interface__]

    # Find and process each satellite image that intersects with the polygon
    for image_filename in os.listdir(image_dir):
        if not image_filename.endswith('.tif'):
            continue
        with rasterio.open(os.path.join(image_dir, image_filename)) as src:
            # Check if the polygon intersects the image
            # Convert the image bounds to a Polygon
            image_bounds = Polygon.from_bounds(*src.bounds)

            # Check if the polygon intersects the image
            if not polygon['geometry'].intersects(image_bounds):
                continue
            else:
                # Mask the image with the polygon to get the area of interest
                out_image, out_transform = mask(src, geo_json, crop=True)
                # Adjust dimensions to be divisible by 32
                out_image_adjusted = out_image#[adjust_dimensions(band) for band in out_image]
                
                """saving binary masked image"""
    
                # Create a binary mask where 1 represents the mining area
                mask_image = rasterio.features.rasterize(
                    [(polygon['geometry'], 1)],
                    out_shape=out_image_adjusted[0].shape,  # Assuming all bands have the same shape
                    transform=out_transform,
                    fill=0,
                    dtype=rasterio.uint8
                )
    
                # Adjust dimensions of the mask to be divisible by 32
                mask_image_adjusted = mask_image#adjust_dimensions(mask_image)
                
                # Calculate new window to read from src using the dimensions of the adjusted mask
                new_height, new_width = mask_image_adjusted.shape
                #new_window = rasterio.windows.Window(col_off=0, row_off=0, width=new_width, height=new_height)
    
                # Calculate the offset for the window based on the adjusted transform
                col_off, row_off = out_transform.c, out_transform.f
                new_window = rasterio.windows.Window(col_off=int(col_off), row_off=int(row_off), width=new_width, height=new_height)
    
                # Read the new window from the source image
                new_out_image = src.read(window=new_window)
    
                with rasterio.open(
                    f'/path', 'w',
                    driver='GTiff',
                    height=mask_image_adjusted.shape[0],
                    width=mask_image_adjusted.shape[1],
                    count=1,
                    dtype=mask_image_adjusted.dtype,
                    crs=src.crs,
                    transform=out_transform
                ) as mask_dest:
                    mask_dest.write(mask_image_adjusted, 1)
                    print(f"done_{index}_mask")
                    
                """saving actual image"""
    
                # Save the new extracted image
                with rasterio.open(
                    f'/path', 'w',
                    driver='GTiff',
                    height=new_height,
                    width=new_width,
                    count=len(new_out_image),
                    dtype=new_out_image[0].dtype,
                    crs=src.crs,
                    transform=out_transform  # Use the transformed coordinates
                ) as dest:
                    for band_index, band in enumerate(new_out_image, start=1):
                        dest.write(band, band_index)
                    print(f"done_{index}_image")
                    

# Approach - 2 (Removed outliers, then resize all the images and masks to dimensions of biggest image )
"""removing outliers"""
import matplotlib.pyplot as plt
Area=[]
# Loop over each polygon in the shapefile
for index, polygon in mining_sites.iterrows():
    # Convert the polygon geometry to GeoJSON format
    geo_json = [polygon['geometry'].__geo_interface__]
    Area.append(polygon['geometry'].area)
Area = np.array(Area)
lower_bound = np.percentile(Area, 10)
upper_bound = np.percentile(Area, 98.5)
print(lower_bound,  upper_bound)
count=len(mining_sites)
for index, polygon in mining_sites.iterrows():
    geo_json = [polygon['geometry'].__geo_interface__]
    polygon_area = polygon['geometry'].area
    if polygon_area < lower_bound or polygon_area > upper_bound:
        count = count-1
print("number of filtered mining sites : ", count)

biggest_shape = None
biggest_image_filename = None
# Loop over each polygon in the shapefile
for index, polygon in mining_sites.iterrows():
    # Convert the polygon geometry to GeoJSON format
    geo_json = [polygon['geometry'].__geo_interface__]
    polygon_area = polygon['geometry'].area
    if polygon_area < lower_bound or polygon_area > upper_bound:
        continue
    # Find and process each satellite image that intersects with the polygon
    for image_filename in os.listdir(image_dir):
        if not image_filename.endswith('.tif'):
            continue
        with rasterio.open(os.path.join(image_dir, image_filename)) as src:
            # Check if the polygon intersects the image
            # Convert the image bounds to a Polygon
            image_bounds = Polygon.from_bounds(*src.bounds)
            if not polygon['geometry'].intersects(image_bounds):
                continue
            else:
                out_image, out_transform = mask(src, geo_json, crop=True)
                current_shape = out_image.shape
                if biggest_shape is None or np.prod(current_shape) > np.prod(biggest_shape):
                    biggest_shape = current_shape
                    biggest_image_filename = index

print(f"Shape of the biggest out_image: {biggest_shape}")
print(f"Filename of the image with the biggest out_image: {biggest_image_filename}")

for index, polygon in mining_sites.iterrows():
    geo_json = [polygon['geometry'].__geo_interface__]
    polygon_area = polygon['geometry'].area
    if polygon_area < lower_bound or polygon_area > upper_bound:
        continue  # Skip this polygon as its area is outside the acceptable range
    for image_filename in os.listdir(image_dir):
        if not image_filename.endswith('.tif'):
            continue
        with rasterio.open(os.path.join(image_dir, image_filename)) as src:
            image_bounds = Polygon.from_bounds(*src.bounds)
            if not polygon['geometry'].intersects(image_bounds):
                continue
            else:
                out_image, out_transform = mask(src, geo_json, crop=True)
                if out_image.shape[1] > biggest_shape[1] or out_image.shape[2] > biggest_shape[2]:
                    print(f"Skipping image {image_filename} due to incompatible shape")
                    continue
                
                # Resize the out_image to match the shape of the biggest one
                pad_width = (
                    (0, 0), 
                    (0, biggest_shape[1] - out_image.shape[1]), 
                    (0, biggest_shape[2] - out_image.shape[2])
                )
                padded_out_image = np.pad(out_image, pad_width, mode='constant')

                mask_image = rasterio.features.rasterize(
                    [(polygon['geometry'], 1)],
                    out_shape=(padded_out_image.shape[1], padded_out_image.shape[2]),
                    # Assuming all bands have the same shape
                    transform=out_transform,
                    fill=0,
                    dtype=rasterio.uint8
                )

                height, width = padded_out_image.shape[1], padded_out_image.shape[2]
                col_off, row_off = out_transform.c, out_transform.f
                new_window = rasterio.windows.Window(col_off=int(col_off), row_off=int(row_off), width=width,
                                                     height=height)
                new_out_image = src.read(window=new_window)

                # Saving the masks
                with rasterio.open(
                        f'/path', 'w',
                        driver='GTiff',
                        height=padded_out_image.shape[1],
                        width=padded_out_image.shape[2],
                        count=1,
                        dtype=padded_out_image.dtype,
                        crs=src.crs,
                        transform=out_transform
                ) as mask_dest:
                    mask_dest.write(mask_image, 1)
                    print(f"done_{index}_mask")

                # Saving the src image
                with rasterio.open(
                        f'/path', 'w',
                        driver='GTiff',
                        height=height,
                        width=width,
                        count=len(new_out_image),
                        dtype=new_out_image[0].dtype,
                        crs=src.crs,
                        transform=out_transform  # Use the transformed coordinates
                ) as dest:
                    for band_index, band in enumerate(new_out_image, start=1):
                        dest.write(band, band_index)
                    print(f"done_{index}_image")


# Approach -3 (Finding the max(height,width) of each mine bound box, and making it a square by adding padding)
import numpy as np

# Function to pad the image to be square and keep the mining site at the center
def pad_image_to_square(image):
    # Get the dimensions of the image
    _, height, width = image.shape
    
    # Calculate the maximum dimension
    max_dim = max(height, width)
    
    # Calculate padding for each side
    pad_height = (max_dim - height) + 50
    pad_width = (max_dim - width) + 50
    
    # Calculate the padding for top/bottom and left/right sides
    pad_top_bottom = pad_height // 2
    pad_left_right = pad_width // 2
    
    # Pad the image
    padded_image = np.pad(image, ((0, 0), (pad_top_bottom, pad_height - pad_top_bottom), (pad_left_right, pad_width - pad_left_right)), mode='constant')
    
    return padded_image


def get_padded_transform(original_transform, padded_width, padded_height):
    # Calculate new transform parameters
    new_c = original_transform.c - (original_transform.a * (padded_width - out_image.shape[2]) / 2)
    new_f = original_transform.f - (original_transform.e * (padded_height - out_image.shape[1]) / 2)
    
    # Create a new Affine object with updated parameters
    padded_transform = rasterio.transform.Affine(original_transform.a, original_transform.b, new_c,
                                                 original_transform.d, original_transform.e, new_f)
    return padded_transform


for index, polygon in mining_sites.iterrows():
    geo_json = [polygon['geometry'].__geo_interface__]
    for image_filename in os.listdir(image_dir):
        if not image_filename.endswith('.tif'):
            continue
        with rasterio.open(os.path.join(image_dir, image_filename)) as src:
            image_bounds = Polygon.from_bounds(*src.bounds)
            if not polygon['geometry'].intersects(image_bounds):
                continue
            else:
                out_image, out_transform = mask(src, geo_json, crop=True)
                padded_out_image = pad_image_to_square(out_image)
                
                _, padded_height, padded_width = padded_out_image.shape
                padded_transform = get_padded_transform(out_transform, padded_width, padded_height)
                mask_image = rasterio.features.rasterize(
                    [(polygon['geometry'], 1)],
                    out_shape=(padded_out_image.shape[1], padded_out_image.shape[2]),  # Assuming all bands have the same shape
                    transform=padded_transform,
                    fill=0,
                    dtype=rasterio.uint8
                )
            
                col_off, row_off = padded_transform.c, padded_transform.f
                new_window = rasterio.windows.Window(col_off=int(col_off), row_off=int(row_off), width = padded_width, height = padded_height)
                new_out_image = src.read(window=new_window)
        

                # Saving the masks
                with rasterio.open(
                    f'/path', 'w',
                    driver='GTiff',
                    height=padded_out_image.shape[1],
                    width=padded_out_image.shape[2],
                    count=1,
                    dtype=padded_out_image.dtype,
                    crs=src.crs,
                    transform=out_transform
                ) as mask_dest:
                    mask_dest.write(mask_image, 1)
                    print(f"done_{index}_mask")
                
                # Saving the src image
                with rasterio.open(
                    f'/path', 'w',
                    driver='GTiff',
                    height=padded_height,
                    width=padded_width,
                    count=len(new_out_image),
                    dtype=new_out_image[0].dtype,
                    crs=src.crs,
                    transform=out_transform  # Use the transformed coordinates
                ) as dest:
                    for band_index, band in enumerate(new_out_image, start=1):
                        dest.write(band, band_index)
                    print(f"done_{index}_image")


