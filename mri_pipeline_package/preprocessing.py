"""
Image preprocessing and augmentation for MRI classification
"""

from torchvision import transforms
import numpy as np
import torch

# Import necessary libraries
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import DefaultDataCollator

# Define normalization parameters
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

# Define size for random resized crop
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)

# Define image transformations
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])


# Import necessary libraries
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import DefaultDataCollator

# Define normalization parameters
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

# Define size for random resized crop
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)

# Define image transformations
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])


# Define data augmentation for training and validation
size = (image_processor.size["height"], image_processor.size["width"])

# Data augmentation for training
train_data_augmentation = keras.Sequential(
    [
        layers.RandomCrop(size[0], size[1]),
        layers.Rescaling(scale=1.0 / 127.5, offset=-1),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="train_data_augmentation",
)

# Data augmentation for validation
val_data_augmentation = keras.Sequential(
    [
        layers.CenterCrop(size[0], size[1]),
        layers.Rescaling(scale=1.0 / 127.5, offset=-1),
    ],
    name="val_data_augmentation",
)


import numpy as np
import tensorflow as tf
from PIL import Image


def convert_to_tf_tensor(image: Image):
    # Convert PIL Image to numpy array and then to TensorFlow tensor
    np_image = np.array(image)
    tf_image = tf.convert_to_tensor(np_image)
    return tf.expand_dims(tf_image, 0)


def preprocess_train(example_batch):
    # Apply data augmentation to training examples
    images = [
        train_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
    ]
    # Transpose and squeeze to match the expected format
    example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
    return example_batch


def preprocess_val(example_batch):
    # Apply data augmentation to validation examples
    images = [
        val_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
    ]
    # Transpose and squeeze to match the expected format
    example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
    return example_batch


