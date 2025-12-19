# parkinsons-mri-classification


This repository provides a reproducible and ethically compliant deep learning framework for classifying brain MRI slices into Healthy and Mild Cognitive Impairment (MCI) categories, corresponding to the prodromal stage of Parkinson’s disease.
The project implements and compares three classification pipelines based on different modeling and preprocessing strategies.
Only minimal, anonymized example images are included to ensure data privacy and responsible data sharing.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

System Flowchart:

![Figure 1](https://github.com/user-attachments/assets/a28fd54f-785c-4a7f-9822-9620910d0677)


-------------------------------------------------------------------------------------------------------------------------------------------------------------------

Methods Overview:  

This study investigates Parkinson’s disease classification using a progressive three-stage experimental design based on the same raw MRI dataset and a consistent evaluation protocol.

• **Method 1 – CNN Ensemble (Baseline)**
  - Input: Raw T1-weighted MRI slices
  - Model: Fine-tuned ensemble of VGG16 and ResNet50
  - Purpose: Establish a baseline without advanced preprocessing

• **Method 2 – FastAI + FSL BET**
  - Input: Same MRI data after full FSL preprocessing with skull stripping (BET)
  - Model: Ensemble-based pipeline retrained using FastAI with ResNet50
  - Strength: Highest recall (0.996), minimizing false negatives

• **Method 3 – FastAI (No BET)**
  - Input: Same MRI data after full FSL preprocessing without skull stripping
  - Model: Ensemble-based pipeline retrained using FastAI with ResNet50
  - Outcome: Best overall performance with highest accuracy (99.61%)
    and F1-score (0.996)

Methods 2 and 3 reuse the same raw MRI dataset and baseline modeling design introduced in Method 1. However, prior to training, the images are processed using FSL, with the presence or absence of BET defining the key distinction between the two methods. This controlled design enables a direct and fair comparison of preprocessing strategies while keeping the core experimental setup unchanged.


| Method | Preprocessing Strategy | TP | FN | TN | FP | Accuracy | Precision | F1-score | Recall |
|--------|------------------------|----|----|----|----|----------|-----------|----------|--------|
| Method 1 | Raw MRI + VGG16 + ResNet50 Ensemble | 480 | 25 | 480 | 35 | 93.05% | 0.932 | 0.941 | 0.951 |
| Method 2 | Full preprocessing (with BET) + Ensemble + FastAI ResNet50 | 538 | 2 | 560 | 18 | 98.21% | 0.968 | 0.982 | 0.996 |
| Method 3 | Full preprocessing (without BET) + Ensemble + FastAI ResNet50 | 512 | 1 | 510 | 3 | 99.61% | 0.994 | 0.996 | 0.998 |

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Preprocessing Clarification

All FSL-based preprocessing steps (including skull stripping with BET) were performed externally prior to model training.
The Python scripts provided in this repository operate exclusively on the resulting 2D MRI slices exported from the FSL pipeline.

No FSL commands are executed within the training scripts.
This design ensures a clear separation between neuroimaging preprocessing and deep learning model development.


-------------------------------------------------------------------------------------------------------------------------------------------------------------------
Repository Structure:

parkinsons-mri-classification/  
│  
├── code/  
│   ├── data_split.py  
│   ├── method1_ensemble.py  
│   ├── method2_fastai.py  
│   ├── method3_fastai.py  
│   └── utils.py  
│  
├── data_samples/  
│   ├── method1/  
│   ├── method2/  
│   └── method3/  
│  
├── preprocessing/  
│   └── FSL_instructions.md  
│  
├── docs/  
│   ├── methodology.md  
│   ├── method1.md  
│   ├── method2.md  
│   └── method3.md  
│  
├── data/  
│   └── README.md  
│  
├── environment.yml  
├── README.md  
├── LICENSE  
└── gitignore  


-------------------------------------------------------------------------------------------------------------------------------------------------------------------
Data Source and Privacy:    

MRI data used in this research originate from the Parkinson’s Progression Markers Initiative (PPMI) database.  
Raw MRI scans and full datasets are not included due to ethical and legal constraints.  
Only a few anonymized MRI slices are provided for demonstration purposes.    
Researchers wishing to reproduce the experiments must obtain the original data directly from the PPMI portal and follow its data-use agreements.  
PPMI: https://www.ppmi-info.org/  

  | Gender | Group | Average Age | Age Range | Number of Individuals | Subject IDs |
|--------|-------|-------------|-----------|------------------------|-------------|
| Female | Control | 55.4 | 40–69 | 8 | 3055, 3106, 3353, 3361, 3569, 3851, 3855, 3857 |
| Male | Control | 59.1 | 44–77 | 12 | 3104, 3301, 3318, 3357, 3369, 3551, 3554, 3563, 3571, 3852, 3853, 3854 |
| Female | MCI | 63.6 | 43–73 | 10 | 40360, 42072, 50572, 51330, 52353, 56126, 60036, 60073, 60074, 60075 |
| Male | MCI | 63.2 | 50–82 | 10 | 16644, 41471, 50110, 50961, 60006, 60024, 60043, 60044, 60091, 72138 |

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

Installation

Install dependencies:
pip install -r env/requirements.txt

Training:  
Method 1 (Ensemble CNN):
python code/method1.py

Method 2 (FastAI + BET):
python code/method2.py


Method 3 (FastAI No BET):
python code/method3.py

## Training and Reproducibility

The training scripts provided in this repository assume access to the full PPMI MRI dataset.
Due to data-sharing restrictions, the complete dataset is not included.

The sample images provided under `data_samples/` are intended for code structure verification and demonstration purposes only and are not sufficient for full model training.

To reproduce the reported results:
1. Obtain the raw MRI data from the PPMI database.
2. Apply the described FSL preprocessing pipeline.
3. Organize the exported MRI slices according to the directory structure described in the documentation.
4. Run the corresponding training script for each method.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

Environment setup:
conda env create -f environment.yml
conda activate parkinsons-mri-classification


-------------------------------------------------------------------------------------------------------------------------------------------------------------------
Results:  

The experimental results reported in the paper show that Method 3
(outlined in this repository) achieved the best performance with an
accuracy of 99.61%.

For full quantitative results, comparisons, and analysis, please refer
to the published article.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------
License  

This project is released under the MIT License. See the LICENSE file for details.  

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

Citation:    
If you use this repository in academic work, please cite the associated manuscript and acknowledge the use of the PPMI dataset according to its data-use policy.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------
