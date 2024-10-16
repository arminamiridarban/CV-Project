import numpy as np
import nibabel as nib
from skimage.morphology import binary_opening, binary_closing, ball, remove_small_objects, binary_dilation
from skimage.measure import label, regionprops, marching_cubes
from mayavi import mlab
import json

file_list = ['001.nii.gz', '003.nii.gz', '025.nii.gz', '046.nii.gz', '062.nii.gz']
final = {}
# Load the NIfTI file (MRI)
for f in file_list:
    file = f
    try:
        nifti_file = 'Task06_Lung/imagesTr/lung_' + file
        ground_truth = 'Task06_Lung/labelsTr/lung_' + file

        img_nifti = nib.load(nifti_file)
        img_data = img_nifti.get_fdata()
        ground_truth_nifti = nib.load(ground_truth)
        ground_truth_data = ground_truth_nifti.get_fdata()

        # Extract voxel dimensions from the affine matrix
        voxel_sizes = np.abs(np.diag(img_nifti.affine)[:3])
        voxel_volume = np.prod(voxel_sizes)  # Volume of a single voxel in mm^3

        #print(f"Voxel dimensions (mm): {voxel_sizes}")
        #print(f"Voxel volume (mm^3): {voxel_volume}")

        # Define tumor size in mm and calculate its volume
        tumor_diameter_mm = 30  # 30 mm diameter
        tumor_volume_mm3 = (4/3) * np.pi * (tumor_diameter_mm / 2) ** 3

        # Convert the tumor volume from mm^3 to voxel space
        max_region_size_voxels = tumor_volume_mm3 / voxel_volume

        #print(f"Tumor volume (mm^3): {tumor_volume_mm3}")
        #print(f"Max region size (in voxels): {max_region_size_voxels}")

        # Step 1: Create a binary mask where -1000 voxels are 0 and all others are 1
        img_data = np.where((img_data >= -100) & (img_data <= 200), img_data, -1000)
        binary_mask = np.where(img_data > -1000, 1, 0)

        # Step 2: Apply binary opening and closing with a 3D structuring element (ball)
        structuring_element = ball(1)  # Adjust ball size based on the level of noise you want to remove
        cleaned_mask = binary_opening(binary_mask, structuring_element)
        cleaned_mask = binary_closing(cleaned_mask, structuring_element)

        # Step 3: Label connected components
        labeled_img = label(cleaned_mask)

        # Step 4: Remove small objects based on the `ball(1)` size (approx. 7 voxels)
        min_region_size_voxels = np.sum(ball(6))  # Volume of a 3D ball with radius 6
        labeled_img = remove_small_objects(labeled_img, min_size=min_region_size_voxels)

        # Step 5: Filter regions based on size (remove regions larger than tumor size)
        for region in regionprops(labeled_img):
            if region.area > max_region_size_voxels:
                labeled_img[labeled_img == region.label] = 0  # Remove the region if it exceeds max size

        # Step 6: Dilate the ground truth to define adjacency (create a binary mask)
        dilated_ground_truth = binary_dilation(ground_truth_data > 0, ball(2))

        # Step 7: Keep only segments that are adjacent to the ground truth
        adjacent_mask = np.logical_and(labeled_img > 0, dilated_ground_truth)

        # Step 8: Apply the adjacency filter back to the original image data
        filtered_mask = labeled_img * adjacent_mask  # Keep only adjacent segments
        img_data[filtered_mask == 0] = -1000  # Set non-adjacent areas to background

        # Save the modified NIfTI image if needed

        if np.any(filtered_mask):
            # Step 9: Convert filtered_mask and ground_truth_data to binary (0 or 1)
            filtered_mask = np.where(filtered_mask > 0, 1, 0)  # Set all non-zero values to 1
            ground_truth_data = np.where(ground_truth_data > 0, 1, 0)  # Set all non-zero values to 1

            # Step 10: Calculate the volume of the filtered segmented regions and the ground truth
            filtered_mask_volume = np.sum(filtered_mask)  # Count voxels in the segmented region
            #print(f"Volume of filtered_mask_volume is {filtered_mask_volume}")
            
            ground_truth_volume = np.sum(ground_truth_data)  # Count voxels in the ground truth region
            #print(f"Volume of ground_truth_volume is {ground_truth_volume}")
            
            # Step 11: Calculate the overlap volume (intersection of segmented and ground truth)
            overlap_volume = np.sum(np.logical_and(filtered_mask, ground_truth_data))  # Intersection of segmented and ground truth
            
            # Step 12: Calculate the overlap ratio (volume of intersection over the volume of the ground truth)
            overlap_ratio = overlap_volume / ground_truth_volume if ground_truth_volume > 0 else 0
            print(f"Overlap ratio (Segmented/GT) of the file {file} is {overlap_ratio*100} %")
            final[file] = overlap_ratio


            # 3D Representation of the Filtered Segmented Regions
            # Extract a 3D mesh from the adjacent segmented regions
            verts, faces, _, _ = marching_cubes(filtered_mask, level=0)
            verts2, faces2, _, _ = marching_cubes(ground_truth_data, level=0)

            # Create a 3D visualization with Mayavi
            mlab.figure(f'Filtered Tumor Segmentation {file}', bgcolor=(0, 0, 0))

            # Plot the segmented tumor in red
            mlab.triangular_mesh([vert[0] * voxel_sizes[0] for vert in verts],
                                [vert[1] * voxel_sizes[1] for vert in verts],
                                [vert[2] * voxel_sizes[2] for vert in verts],
                                faces,
                                color=(1, 0, 0),
                                opacity=0.3)

            # Plot the ground truth in green
            mlab.triangular_mesh([vert[0] * voxel_sizes[0] for vert in verts2],
                                [vert[1] * voxel_sizes[1] for vert in verts2],
                                [vert[2] * voxel_sizes[2] for vert in verts2],
                                faces2,
                                color=(0, 1, 0),
                                opacity=0.3)

            # Show the 3D plot
            mlab.show()
            """
            output_file = f"modified_{file}"
            modified_nifti = nib.Nifti1Image(img_data.astype(np.float32), affine=img_nifti.affine, header=img_nifti.header)
            nib.save(modified_nifti, output_file)
            print(f"Filtered NIfTI image saved as {output_file}")
            """
        else:
            continue

    except Exception as e:
        continue

with open('x.json', 'w') as f:
    json.dump(final, f)




        