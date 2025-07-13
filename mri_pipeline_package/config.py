"""
Configuration settings for MRI Tumor Classification Pipeline
"""

import os
from pathlib import Path

# Dataset Configuration
DATASET_PATH = "./mri_data_full"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")

# Model Configuration
MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"
MODEL_NAME = "mri_tumor_classifier"
NUM_CLASSES = 2
IMAGE_SIZE = 224

# Training Configuration
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 0

# Data Augmentation Configuration
RANDOM_ROTATION = 0.02
RANDOM_ZOOM = 0.2
RANDOM_FLIP_PROB = 0.5

# Visualization Configuration
FIGURE_SIZE = (15, 10)
HISTOGRAM_BINS = 256
GRID_SIZE = (3, 3)

# Labels
LABELS = ["notumor", "tumor"]
LABEL_COLORS = {
    "notumor": "#2E8B57",  # Sea Green
    "tumor": "#DC143C"     # Crimson
}

# Streamlit Configuration
STREAMLIT_TITLE = "🧠 MRI Tumor Classification"
STREAMLIT_DESCRIPTION = """
This application uses a Vision Transformer (ViT) to classify MRI brain scans 
as either containing a tumor or not. Upload your MRI image to get instant classification results.
"""

# Font Configuration
FONT_FAMILY = "Tw Cen MT"
FONT_SIZE = 14 