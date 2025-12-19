"""
Method 1: CNN Ensemble (VGG16 + ResNet50)
----------------------------------------
Parkinson's Disease classification (MCI vs Healthy)
using raw T1-weighted MRI slices (no skull stripping).

This script implements the baseline ensemble model used in our experiments for binary classification of Healthy vs. MCI subjects based on raw T1-weighted MRI slices.

The model combines fine-tuned VGG16 and ResNet50 backbones and serves as the reference framework for evaluating the impact of preprocessing and alternative training strategies in subsequent methods.

The implementation is provided to ensure full reproducibility of the reported experimental results.

# NOTE:
# Training on full 512x512 slices significantly increases memory usage.
# Depending on the available GPU, reducing the input resolution may be required.

"""


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.layers import Dense, Dropout, Flatten, Average, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix


# Configuration
DATA_DIR = "data/method1_raw_split"   # train/validation/test/healthy,mci
OUTPUT_DIR = "outputs/method1_ensemble"

IMG_SIZE = (512, 512)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data Generators
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.01,
    height_shift_range=0.01
)

train_gen = datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = datagen.flow_from_directory(
    os.path.join(DATA_DIR, "validation"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_gen = datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


# VGG16 Branch
vgg_base = VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMG_SIZE, 3)
)

# Freeze early convolutional layers to reduce overfitting
for layer in vgg_base.layers[:16]:
    layer.trainable = False

# VGG branch: slightly higher capacity
vgg = Flatten()(vgg_base.output)
vgg = Dense(256, activation="relu", kernel_regularizer=l2(0.01))(vgg)
vgg = Dropout(0.5)(vgg)
vgg = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(vgg)
vgg = Dropout(0.5)(vgg)
vgg_out = Dense(2, activation="softmax")(vgg)

model_vgg = Model(vgg_base.input, vgg_out)


# ResNet50 Branch
res_base = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMG_SIZE, 3)
)

# Freeze most of the ResNet50 layers; fine-tune only the top blocks
for layer in res_base.layers[:169]:
    layer.trainable = False
    
# ResNet branch: lower capacity to reduce redundancy
res = Flatten()(res_base.output)
res = Dense(256, activation="relu", kernel_regularizer=l2(0.01))(res)
res = Dropout(0.5)(res)
res = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(res)
res = Dropout(0.5)(res)
res_out = Dense(2, activation="softmax")(res)

model_res = Model(res_base.input, res_out)


# Ensemble Model
ensemble_input = Input(shape=(*IMG_SIZE, 3))
out_vgg = model_vgg(ensemble_input)
out_res = model_res(ensemble_input)

ensemble_output = Average()([out_vgg, out_res])
ensemble_model = Model(ensemble_input, ensemble_output)

ensemble_model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

ensemble_model.summary()


# Training
checkpoint = ModelCheckpoint(
    os.path.join(OUTPUT_DIR, "ensemble_vgg16_resnet50_rawMRI.h5"),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

history = ensemble_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint],
    shuffle=True
)


# Evaluation on Test Set
test_gen.reset()
preds = ensemble_model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)

print("\nClassification Report (Test Set):")
print(classification_report(test_gen.classes, y_pred))

print("Confusion Matrix (Test Set):")
# Performance evaluation on the held-out test set
print("\nTest set performance:")
print(classification_report(
    test_gen.classes,
    y_pred,
    target_names=list(test_gen.class_indices.keys())
))
