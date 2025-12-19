# Method 1: CNN Ensemble (Baseline)

## Overview

Method 1 serves as the baseline model for this study. It performs classification directly on raw T1-weighted MRI slices without advanced preprocessing such as skull stripping.

## Input Data

- Raw MRI slices
- No FSL preprocessing
- Two classes: Healthy and MCI

The dataset is organized into class-specific folders without explicit train/validation/test directories.

## Model Architecture

An ensemble of two pretrained convolutional neural networks is used:

- VGG16
- ResNet50

Both networks are initialized with ImageNet weights and partially fine-tuned. Their outputs are combined using an average layer to form the final prediction.

## Training Strategy

Data splitting is performed internally using the validation split mechanism of the Keras `ImageDataGenerator`. This approach enables training and validation without physically separating images into different directories.

## Purpose

This method establishes a reference performance level and provides a baseline for assessing the benefits of preprocessing and alternative training frameworks introduced in subsequent methods.
