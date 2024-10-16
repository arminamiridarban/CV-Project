import json
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.filters import gaussian
from skimage.morphology import opening, closing, disk
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from sklearn.metrics import jaccard_score, f1_score

# Load the JSON file to get the specific slice number for lung_003
with open('Layers.json', 'r') as f:
    layers_data = json.load(f)

# File paths for the image and ground truth
nifti_file = 'Task06_Lung/imagesTr/lung_014.nii.gz'
ground_truth_file = 'Task06_Lung/labelsTr/lung_014.nii.gz'

file_name = os.path.basename(nifti_file)
file_id = file_name.split('.')[0]

# Use the extracted file identifier to get the slice index from JSON
slice_idx = layers_data.get(file_id)

# Load the NIfTI files
img_nifti = nib.load(nifti_file)
ground_truth_nifti = nib.load(ground_truth_file)

# Extract image data and ground truth data
img_data = img_nifti.get_fdata()
ground_truth = ground_truth_nifti.get_fdata()

# Function to preprocess image slice
def preprocess_slice(slice_data):
    slice_data = slice_data - np.min(slice_data)
    slice_data = slice_data / np.max(slice_data)
    slice_contrast = exposure.equalize_adapthist(slice_data)
    slice_blur = gaussian(slice_contrast, sigma=1)
    return slice_blur

# Function to segment tumor in a given slice
def segment_tumor(slice_data, hu_min=0, hu_max=100, min_tumor_area=50, max_tumor_area=5000):
    hu_filtered_mask = np.logical_and(slice_data > hu_min, slice_data < hu_max)
    if np.any(hu_filtered_mask):
        # Using 20% of upper threshold of HU
        threshold = np.percentile(slice_data[hu_filtered_mask], 80)
    else:
        return np.zeros_like(slice_data)

    binary_mask = np.logical_and(slice_data > threshold, hu_filtered_mask)
    selem = disk(1)
    cleaned_mask = opening(binary_mask, selem)
    cleaned_mask = closing(cleaned_mask, selem)
    labeled_mask = label(cleaned_mask)
    tumor_mask = np.zeros_like(cleaned_mask)

    for region in regionprops(labeled_mask):
        if min_tumor_area < region.area < max_tumor_area:
            if region.eccentricity < 0.95 and region.solidity > 0.8:
                tumor_mask[labeled_mask == region.label] = 1

    final_mask = binary_fill_holes(tumor_mask)
    return final_mask

# Function to display a specific slice
def display_slice(slice_idx):
    slice_data = img_data[:, :, slice_idx]
    ground_truth_slice = ground_truth[:, :, slice_idx]
    
    # Preprocess the slice
    slice_processed = preprocess_slice(slice_data)
    
    # Perform tumor segmentation
    tumor_mask = segment_tumor(slice_processed)
    
    # Calculate Jaccard and Dice indices
    flat_ground_truth = ground_truth_slice.flatten() > 0
    flat_tumor_mask = tumor_mask.flatten()
    
    jaccard_index = jaccard_score(flat_ground_truth, flat_tumor_mask)
    dice_index = f1_score(flat_ground_truth, flat_tumor_mask)
    
    # Calculate the average HU value of the current slice
    avg_hu_value = np.mean(slice_data[np.logical_and(slice_data > -1000, slice_data < 1000)])
    
    # Display the results
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(slice_data, cmap='gray')
    ax[0].set_title(f'Slice {slice_idx} - Original (Avg HU: {avg_hu_value:.2f})')
    
    ax[1].imshow(tumor_mask, cmap='gray')
    ax[1].set_title(f'Slice {slice_idx} - Tumor Segmentation')
    
    ax[2].imshow(ground_truth_slice, cmap='gray')
    ax[2].set_title(f'Slice {slice_idx} - Ground Truth (Jaccard: {jaccard_index:.4f}, Dice: {dice_index:.4f})')
    
    plt.show()

# Display the slice mentioned for lung_003 in the JSON
display_slice(slice_idx)
