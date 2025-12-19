# Method 2: FastAI + FSL BET

## Overview

Method 2 extends the baseline by introducing a full MRI preprocessing pipeline using FSL, including skull stripping via the Brain Extraction Tool (BET). Model training is performed using the FastAI framework.

## Preprocessing Pipeline

- Bias field correction
- Spatial normalization
- Skull stripping using FSL BET
- Slice extraction from relevant brain regions

The output images from FSL preprocessing are used as input for model training.

## Data Organization

Unlike Method 1, this method requires an explicit folder-based split into training, validation, and test sets. The dataset is organized as:

- train/
- validation/
- test/

Each split contains separate folders for Healthy and MCI classes.

## Model and Training

- Backbone: ResNet50
- Framework: FastAI
- Training includes data augmentation and fine-tuning

## Motivation

The inclusion of skull stripping aims to remove non-brain tissues and improve model focus on disease-relevant regions. This method achieves the highest recall, making it particularly suitable for minimizing false negatives in clinical screening scenarios.
