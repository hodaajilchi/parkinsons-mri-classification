# Methodology

This study proposes a unified experimental framework for the classification of Parkinson’s disease at the prodromal stage using T1-weighted brain MRI slices. All experiments are conducted on the same underlying dataset and follow a consistent evaluation protocol to ensure fair comparison across different modeling strategies.

The methodology consists of three sequential methods that differ in their preprocessing pipelines and training frameworks, while preserving the same classification objective and label definitions (Healthy vs.MCI).

## Dataset Preparation

All MRI data originate from the Parkinson’s Progression Markers Initiative (PPMI) database. Raw DICOM images were converted to NIfTI format and preprocessed using the FSL software package. Slice selection was restricted to anatomically relevant regions associated with Parkinson’s disease.

Due to ethical and privacy constraints, raw MRI data are not distributed in this repository. Only a limited number of anonymized sample slices are provided for demonstration purposes.

## Experimental Design

- **Method 1** establishes a baseline using raw MRI slices and a CNN ensemble implemented in TensorFlow/Keras.
- **Method 2** applies full FSL preprocessing including skull stripping (BET), followed by training with the FastAI framework.
- **Method 3** applies the same preprocessing pipeline as Method 2, except that skull stripping is omitted.

All methods share the same label definitions, dataset composition, and evaluation metrics. Differences in performance therefore reflect the impact of preprocessing choices and training frameworks rather than dataset variations.
In Method 1, dataset splitting was performed internally using the validation split mechanism of the Keras ImageDataGenerator, without creating separate directories on disk. In contrast, Methods 2 and 3 require explicit folder-based splits due to the FastAI data loading pipeline.
## Evaluation Metrics

Model performance is evaluated using accuracy, precision, recall, and F1-score, along with confusion matrix analysis. Particular emphasis is placed on recall due to its clinical importance in minimizing false negatives.
