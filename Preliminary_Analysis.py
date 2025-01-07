import os
import rasterio as rio
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_path = '/path'
img_path=[]
for file_name in os.listdir(train_path):
            if file_name.startswith("train_") and file_name.endswith(".tif"):
                img_path.append(os.path.join(train_path,file_name))
img_path

def pct_clip(array,pct=[2,98]):
    array_min, array_max = np.nanpercentile(array,pct[0]), np.nanpercentile(array,pct[1])
    clip = (array - array_min) / (array_max - array_min)
    clip[clip>1]=1
    clip[clip<0]=0
    return clip

def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

normalized_images = []
for path in img_path:
    with rio.open(path) as src:
        image = src.read()
        normalized_image = np.zeros_like(image)
        for band in range(image.shape[0]):
            normalized_image[band] = normalize(image[band])
        
        normalized_images.append(normalized_image)

print("total images: ", len(normalized_images))
plt.imshow((normalized_images[2][5]))

def calculate_ndvi(nir, red):
    return (nir - red) / (nir + red)

def calculate_ndwi(nir, green):
    return (nir - green) / (nir + green)

def calculate_ndmi(nir, swir1):
    return (nir - swir1) / (nir + swir1)

def calculate_evi(nir, red, blue):
    return 2.5 * (nir - red) / ((nir + 6 * red - 7.5 * blue) + 1)

from tqdm import tqdm
import itertools
import numpy as np

# Assuming normalized_images is your list of images
enhanced_images = []
for image in tqdm(normalized_images, desc="Processing Images"):
    # Initialize a list to hold the new bands
    new_bands = []
    
    # Iterate over all unique combinations of the 12 bands
    for i, j in itertools.combinations(range(12), 2):
        new_band = (image[i] - image[j]) / (np.clip(image[i] + image[j], a_min=1e-5, a_max=None))
        new_bands.append(new_band[np.newaxis, :])

    # Stack new bands with the original ones
    enhanced_image = np.vstack((image, *new_bands))
    enhanced_images.append(enhanced_image)

print("total images: ", len(enhanced_images))
plt.imshow((enhanced_images[2][4]))

print("total images: ", len(normalized_images))
plt.imshow((normalized_images[2][4]))

# Create an empty list to store the averages
band_averages = []

# Loop over each enhanced image
for enhanced_image in enhanced_images:
    # Calculate the mean of each band
    averages = [np.mean(band) for band in enhanced_image]
    band_averages.append(averages)

# Create a DataFrame from the averages
column_names = [f'Band {i+1}' for i in range(12)] + ['NDVI', 'NDWI', 'NDMI', 'EVI']
average_bands = pd.DataFrame(band_averages, columns=column_names)

# Show the DataFrame shape to confirm it has the right dimensions (1242 rows, 16 columns)
print(average_bands.shape)

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


answer = pd.read_csv("/path")
answer.set_index(answer.columns[0], inplace=True)
# Assuming 'answer' is your target DataFrame and it's aligned with 'average_bands'
y = answer['target']  # Replace 'target' with the actual column name in your 'answer' DataFrame
X = average_bands

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the LightGBM model
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy}')

# Get and print feature importances
feature_importances = model.feature_importances_
for i, importance in enumerate(feature_importances):
    print(f'Feature: {X.columns[i]}, Importance: {importance}')


