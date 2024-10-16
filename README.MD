# Lung Tumor Segmentation using NIfTI Images

This project implements multiple methods for segmenting lung tumors from medical images (NIfTI format), focusing on preprocessing, segmentation, and evaluation techniques. The project uses thresholding, morphological operations, and machine learning metrics to segment and validate lung tumors.

## Features
- **Preprocessing**: Normalization, contrast enhancement, and noise reduction.
- **Segmentation Methods**:
  - **3D Segmentation** using the Marching Cubes algorithm.
  - **HU-based 2D Segmentation** for specific slices with morphological filtering.
- **Evaluation Metrics**: Jaccard Index (Intersection over Union) and Dice Coefficient (F1 Score).
- **Visualization**: 3D rendering of segmented lungs and tumor regions using Mayavi.

## Installation

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Preprocessing and Segmentation**:
   Run the Python scripts to preprocess and segment the NIfTI lung images.

2. **Displaying a Specific Slice**:
   The project includes functionality to display specific slices of the 3D image, apply segmentation, and calculate evaluation metrics like Jaccard and Dice scores.

3. **3D Visualization**:
   The project uses Mayavi to visualize segmented lung regions and tumor masks in 3D.

## Example

To run the segmentation and visualization on a sample file:

```bash
python <filename>.py
```
If you use MacOS, please run the following command:
```bash
python3 <filename>.py
```
