"""
Method 2: FastAI + FSL BET

This script trains a ResNet50-based classifier using the FastAI framework on MRI slices that have been fully preprocessed using FSL, including skull stripping with the Brain Extraction Tool (BET).

All preprocessing steps (registration, normalization, and BET) are performed externally using FSL. This implementation focuses exclusively on model training and evaluation using the preprocessed data.

The method builds upon the same dataset and classification objective introduced in Method 1 and is provided to ensure reproducibility of the reported experimental results.
"""

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

from fastai.vision.all import *
from pathlib import Path

DATA_DIR = Path("data/Method2_FSL_BET_split")
OUTPUT_DIR = Path("outputs/method2_fastai")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def label_func(fname):
    fname = fname.name.lower()  
    if 'mci' in str(fname):
        return 'mci'
    elif 'healthy' in str(fname):
        return 'healthy'
    else:
        return 'other'

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# DataLoaders
dls = ImageDataLoaders.from_folder(
    DATA_DIR,
    train="train",
    valid="validation",
    item_tfms=Resize(224),
    batch_tfms=aug_transforms(mult=2),
    bs=64
)


# Model
learn = cnn_learner(
    dls,
    resnet50,
    metrics=accuracy
)

learn.fine_tune(15)

# Evaluation
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# Save Model
learn.export(OUTPUT_DIR / "method2_fastai.pkl")

accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average='weighted')
recall = recall_score(true_labels, pred_labels, average='weighted')
f1 = f1_score(true_labels, pred_labels, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}'
