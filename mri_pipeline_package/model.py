"""
Vision Transformer training and evaluation pipeline
"""

import torch
from transformers import ViTForImageClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

!tensorboard dev upload --logdir logs --name "ViT MRI classifier" --description "Training the MRI tumor datasets"

# Load necessary libraries
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define axis labels for the confusion matrix
x_axis_labels = ["notumor", "tumor"]
y_axis_labels = ["notumor", "tumor"]

# Compute the confusion matrix
cnf = confusion_matrix(act_op, pred_op)

# Plot the confusion matrix using seaborn
ax = sns.heatmap(cnf, annot=True, cmap='Blues', xticklabels=x_axis_labels, yticklabels=y_axis_labels)


