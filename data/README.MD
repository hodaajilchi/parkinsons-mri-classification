# Data Description and Access

This directory is intentionally left empty and does not contain any MRI data.

## Data Source

The MRI data used in this project originate from the  
**Parkinson’s Progression Markers Initiative (PPMI)** database.

PPMI is a large, multi-center, longitudinal study designed to identify
biomarkers of Parkinson’s disease progression.

Website: https://www.ppmi-info.org/

## Data Availability and Ethics

Due to ethical, legal, and privacy constraints, the original MRI scans
cannot be redistributed publicly.

In accordance with PPMI data-use agreements:
- Raw MRI data
- Preprocessed volumes
- Subject-identifiable information

are **not included** in this repository.

## How to Reproduce the Experiments

Researchers who wish to reproduce the experiments should:

1. Register and request access to the PPMI database via the official portal.
2. Download the required T1-weighted MRI scans. The original MRI data were downloaded from the PPMI database in DICOM format and converted to NIfTI format prior to FSL-based preprocessing.
3. Organize the data according to the structure described below.
4. Apply preprocessing using FSL as described in `preprocessing/FSL_instructions.md`.
5. Run the data splitting and training scripts provided in the `code/` directory.

## Expected Directory Structure

After obtaining and preprocessing the data, the directory structure
should follow this format:


  
data/  
├── raw/  
│ ├── healthy/  
│ └── mci/  
│  
├── Method2_FSL_BET_split/  
│ ├── train/  
│ ├── validation/    
│ └── test/  
│  
├── Method3_FSL_NoBET_split/  
│ ├── train/  
│ ├── validation/  
│ └── test/  

data_split.py: Utility script for organizing MRI slice images into train, validation, and test subsets with fixed class labels (healthy, mci).

## Notes

Only a minimal number of anonymized example images are provided in
`data_samples/` for demonstration and visualization purposes.
These samples are **not used** for training or evaluation.
