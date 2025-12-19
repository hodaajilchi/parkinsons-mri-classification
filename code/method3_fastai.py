"""
Method 3: FastAI without BET

This script trains a ResNet50-based classifier using the FastAI framework on MRI slices preprocessed with FSL, where skull stripping (BET) is intentionally omitted.

All FSL preprocessing steps, including registration and normalization, are performed externally. This script assumes that the resulting 2D MRI slices are already organized into train and validation directories.

This method follows the same dataset organization and evaluation protocol as Method 2, enabling a controlled comparison of the effect of skull stripping on Parkinson’s disease classification performance.

The implementation is provided to ensure reproducibility of the reported experimental results.
"""


from fastai.vision.all import *
from pathlib import Path


# Configuration
DATA_DIR = Path("data/Method3_FSL_NoBET_split")
OUTPUT_DIR = Path("outputs/method3_fastai")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

categories = ['healthy', 'mci']

# Define a function for labeling based on file path
def label_func(fname):
    fname = fname.name.lower()  
    if 'mci' in str(fname):
        return 'mci'
    elif 'healthy' in str(fname):
        return 'healthy'
    else:
        return 'other'

# Ratios
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
learn.export(OUTPUT_DIR / "method3_fastai.pkl")

accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average='weighted')
recall = recall_score(true_labels, pred_labels, average='weighted')
f1 = f1_score(true_labels, pred_labels, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}'
