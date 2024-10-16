import numpy as np
import nibabel as nib
from skimage import filters, measure, morphology
from mayavi import mlab
import os

# Load the NIfTI file
nifti_file = nib.load('Task06_Lung/imagesTr/lung_025.nii.gz')

# Get the image data as a NumPy array
image_data = nifti_file.get_fdata()

# Preprocess the data: Normalize to 0-1 range
image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

# Segment the lungs using a threshold
threshold = filters.threshold_li(image_data)
binary_mask = image_data < threshold
binary_mask = morphology.remove_small_objects(binary_mask.astype(bool), min_size=1000)
binary_mask = morphology.binary_closing(binary_mask, morphology.ball(5))

# Extract the region of interest from the binary mask
def extract_lung_region(binary_mask):
    lung_region = np.zeros_like(binary_mask)
    for z in range(binary_mask.shape[2]):
        contours = measure.find_contours(binary_mask[:, :, z], 0.5)
        for contour in contours:
            if np.all(contour[:, 1] > 10) and np.all(contour[:, 1] < 450) and np.all(contour[:, 0] > 50) and np.all(contour[:, 0] < 450):
                rr, cc = contour.astype(int).T
                lung_region[rr, cc, z] = 1
    return lung_region

# Extract the lung region
lung_region = extract_lung_region(binary_mask)

lung_region_nb = nib.Nifti1Image(lung_region.astype(np.uint8), affine=np.eye(4))

# Extract the surface mesh using marching cubes
verts, faces, normals, values = measure.marching_cubes(lung_region, level=0.5)

# Create a 3D plot using mayavi
mlab.figure('Lung Segmentation', bgcolor=(0, 0, 0))
mlab.triangular_mesh([vert[0] for vert in verts],
                     [vert[1] for vert in verts],
                     [vert[2] for vert in verts],
                     faces,
                     color=(1, 0, 0),
                     opacity=0.1)
mlab.show()
