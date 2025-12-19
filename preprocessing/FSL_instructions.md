# FSL Preprocessing Pipeline for MRI Analysis

This document describes the FSL-based preprocessing steps applied to T1-weighted MRI scans prior to deep learning–based classification.

**Note:**  
FSL preprocessing was applied **only in Methods 2 and 3**.  
**Method 1 did not use FSL preprocessing** and operated on manually selected raw 2D MRI slices.

---
<img width="1201" height="835" alt="FSL GITHUB FINAL" src="https://github.com/user-attachments/assets/cd65760d-b790-401a-8e50-a28f185e9814" />


## Input Data

- Raw T1-weighted MRI scans  
- Format: NIfTI (`.nii` or `.nii.gz`)
- 
## Image Reorientation and Spatial Normalization

All MRI volumes were reoriented to standard anatomical orientation using `fslreorient2std` and subsequently registered to the MNI152 template using FLIRT. This step ensured consistent anatomical alignment across subjects and enabled atlas-guided region-of-interest (ROI) identification.


## Brain Extraction Using BET (Method 2 Only)

Skull stripping was performed **only in Method 2** using FSL’s Brain Extraction Tool (BET) to remove non-brain tissues such as the skull and scalp.

### Command Used

```bash
bet input.nii.gz output_brain.nii.gz -f 0.3 -g 0
```

### Parameter Description

input.nii.gz: raw MRI volume

output_brain.nii.gz: skull-stripped MRI volume

-f 0.3: fractional intensity threshold (selected empirically)

-g 0: vertical gradient correction

This step resulted in a clean brain-only volume suitable for further processing.



## Atlas-Guided ROI Masking and Slice Selection

After registration to MNI space, atlas-guided ROI masks corresponding to Parkinson’s disease–related brain regions (including the substantia nigra, basal ganglia, thalamus, and motor cortex) were applied.

Axial slices intersecting these ROIs were identified and extracted:
Methods 2 and 3: slices 39–100
Method 1: slices 135–165 (manual selection on raw images)

This strategy ensured consistent anatomical coverage across subjects while reducing irrelevant background information. 


<img width="1063" height="209" alt="Figure 7" src="https://github.com/user-attachments/assets/4ffea66e-b7e0-4cbb-bbb0-aefd3fbee7dc" />


## Slice Extraction and Output Preparation

Selected axial slices were extracted from the 3D volumes and saved as individual 2D images (JPG format).

While preprocessing and anatomical alignment were conducted in 3D volumetric space, the final classification was performed using 2D axial slices as input to the CNN models.


## Summary

FSL preprocessing was applied only in Methods 2 and 3.

BET was applied only in Method 2.

Method 3 excluded skull stripping to evaluate the diagnostic contribution of extracranial features.

Atlas-guided ROI-based slice selection was used for Methods 2 and 3.

Slice-based 2D images were used for CNN training and evaluation.


