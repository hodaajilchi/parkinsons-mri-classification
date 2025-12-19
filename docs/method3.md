# Method 3: FastAI without BET

## Overview

Method 3 follows the same preprocessing and training framework as Method2, with the key difference that skull stripping is not applied.

## Preprocessing Pipeline

- Bias field correction
- Spatial normalization
- No skull stripping (BET omitted)
- Slice extraction identical to Method 2

This design isolates the effect of skull stripping while keeping all other factors constant.

## Data Organization

The dataset structure and splitting strategy are identical to Method 2, ensuring a controlled comparison.

## Model and Training

- Backbone: ResNet50
- Framework: FastAI
- Same training parameters and evaluation protocol as Method 2

## Outcome

This method achieves the best overall performance in terms of accuracy and F1-score. The results suggest that, for this dataset, retaining skull information may preserve discriminative features relevant to Parkinson’s disease classification.
