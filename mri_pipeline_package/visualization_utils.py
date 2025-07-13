"""
Visualization utilities for MRI tumor classification
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
from PIL import Image
import config

class MRIVisualizer:
    """Handles visualization of MRI data and results"""
    
    def __init__(self):
        # Set font family
        plt.rcParams['font.family'] = config.FONT_FAMILY
        plt.rcParams['font.size'] = config.FONT_SIZE
        
    def plot_sample_images(self, samples, title="Sample MRI Images"):
        """Plot sample images in a grid"""
        if not samples:
            return None
            
        fig, axes = plt.subplots(3, 3, figsize=config.FIGURE_SIZE)
        fig.suptitle(title, fontsize=16, fontfamily=config.FONT_FAMILY)
        
        for i, sample in enumerate(samples):
            row = i // 3
            col = i % 3
            
            # Convert PIL image to numpy array
            img_array = np.array(sample['image'])
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            axes[row, col].imshow(img_array, cmap='gray')
            axes[row, col].set_title(
                f"{sample['label_name'].title()}", 
                fontfamily=config.FONT_FAMILY,
                color=config.LABEL_COLORS.get(sample['label_name'], 'black')
            )
            axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_class_distribution(self, distribution):
        """Plot class distribution as bar chart"""
        if not distribution:
            return None
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Train distribution
        train_labels = list(distribution['train'].keys())
        train_counts = list(distribution['train'].values())
        colors = [config.LABEL_COLORS.get(label, '#666666') for label in train_labels]
        
        ax1.bar(train_labels, train_counts, color=colors)
        ax1.set_title('Training Set Distribution', fontfamily=config.FONT_FAMILY)
        ax1.set_ylabel('Number of Images', fontfamily=config.FONT_FAMILY)
        
        # Test distribution
        test_labels = list(distribution['test'].keys())
        test_counts = list(distribution['test'].values())
        colors = [config.LABEL_COLORS.get(label, '#666666') for label in test_labels]
        
        ax2.bar(test_labels, test_counts, color=colors)
        ax2.set_title('Test Set Distribution', fontfamily=config.FONT_FAMILY)
        ax2.set_ylabel('Number of Images', fontfamily=config.FONT_FAMILY)
        
        plt.tight_layout()
        return fig
    
    def plot_histogram_comparison(self, tumor_image, non_tumor_image):
        """Plot histogram comparison between tumor and non-tumor images"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Convert images to numpy arrays
        tumor_array = np.array(tumor_image)
        non_tumor_array = np.array(non_tumor_image)
        
        # Convert to grayscale if needed
        if len(tumor_array.shape) == 3:
            tumor_array = cv2.cvtColor(tumor_array, cv2.COLOR_RGB2GRAY)
        if len(non_tumor_array.shape) == 3:
            non_tumor_array = cv2.cvtColor(non_tumor_array, cv2.COLOR_RGB2GRAY)
        
        # Plot histograms
        ax1.hist(tumor_array.ravel(), bins=config.HISTOGRAM_BINS, alpha=0.7, 
                color=config.LABEL_COLORS['tumor'], label='Tumor')
        ax1.set_title('Tumor Image Histogram', fontfamily=config.FONT_FAMILY)
        ax1.set_xlabel('Pixel Intensity', fontfamily=config.FONT_FAMILY)
        ax1.set_ylabel('Frequency', fontfamily=config.FONT_FAMILY)
        
        ax2.hist(non_tumor_array.ravel(), bins=config.HISTOGRAM_BINS, alpha=0.7,
                color=config.LABEL_COLORS['notumor'], label='No Tumor')
        ax2.set_title('Non-Tumor Image Histogram', fontfamily=config.FONT_FAMILY)
        ax2.set_xlabel('Pixel Intensity', fontfamily=config.FONT_FAMILY)
        ax2.set_ylabel('Frequency', fontfamily=config.FONT_FAMILY)
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_result(self, image, prediction_result):
        """Plot image with prediction result"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        ax1.imshow(img_array, cmap='gray')
        ax1.set_title('Input MRI Image', fontfamily=config.FONT_FAMILY)
        ax1.axis('off')
        
        # Prediction results
        if prediction_result:
            label = prediction_result['predicted_label']
            confidence = prediction_result['confidence']
            probabilities = prediction_result['probabilities']
            
            # Color based on prediction
            color = config.LABEL_COLORS.get(label, '#666666')
            
            # Bar chart of probabilities
            labels = config.LABELS
            ax2.bar(labels, probabilities, color=[config.LABEL_COLORS.get(l, '#666666') for l in labels])
            ax2.set_title(f'Prediction: {label.title()}\nConfidence: {confidence:.2%}', 
                         fontfamily=config.FONT_FAMILY, color=color)
            ax2.set_ylabel('Probability', fontfamily=config.FONT_FAMILY)
            ax2.set_ylim(0, 1)
            
            # Add value labels on bars
            for i, v in enumerate(probabilities):
                ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', fontfamily=config.FONT_FAMILY)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_plot(self, samples):
        """Create interactive plot using Plotly"""
        if not samples:
            return None
            
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[f"{sample['label_name'].title()}" for sample in samples],
            specs=[[{"secondary_y": False}] * 3] * 3
        )
        
        for i, sample in enumerate(samples):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            # Convert image to numpy array
            img_array = np.array(sample['image'])
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            fig.add_trace(
                go.Image(z=img_array, colorscale='gray'),
                row=row, col=col
            )
        
        fig.update_layout(
            title_text="Interactive MRI Sample Images",
            font=dict(family=config.FONT_FAMILY, size=config.FONT_SIZE),
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_training_metrics(self, metrics_history):
        """Plot training metrics over epochs"""
        if not metrics_history:
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(1, len(metrics_history['train_loss']) + 1)
        
        # Training loss
        axes[0, 0].plot(epochs, metrics_history['train_loss'], 'b-', label='Training Loss')
        axes[0, 0].set_title('Training Loss', fontfamily=config.FONT_FAMILY)
        axes[0, 0].set_xlabel('Epoch', fontfamily=config.FONT_FAMILY)
        axes[0, 0].set_ylabel('Loss', fontfamily=config.FONT_FAMILY)
        axes[0, 0].legend()
        
        # Validation loss
        if 'val_loss' in metrics_history:
            axes[0, 1].plot(epochs, metrics_history['val_loss'], 'r-', label='Validation Loss')
            axes[0, 1].set_title('Validation Loss', fontfamily=config.FONT_FAMILY)
            axes[0, 1].set_xlabel('Epoch', fontfamily=config.FONT_FAMILY)
            axes[0, 1].set_ylabel('Loss', fontfamily=config.FONT_FAMILY)
            axes[0, 1].legend()
        
        # Accuracy
        if 'accuracy' in metrics_history:
            axes[1, 0].plot(epochs, metrics_history['accuracy'], 'g-', label='Accuracy')
            axes[1, 0].set_title('Accuracy', fontfamily=config.FONT_FAMILY)
            axes[1, 0].set_xlabel('Epoch', fontfamily=config.FONT_FAMILY)
            axes[1, 0].set_ylabel('Accuracy', fontfamily=config.FONT_FAMILY)
            axes[1, 0].legend()
        
        # F1 Score
        if 'f1' in metrics_history:
            axes[1, 1].plot(epochs, metrics_history['f1'], 'm-', label='F1 Score')
            axes[1, 1].set_title('F1 Score', fontfamily=config.FONT_FAMILY)
            axes[1, 1].set_xlabel('Epoch', fontfamily=config.FONT_FAMILY)
            axes[1, 1].set_ylabel('F1 Score', fontfamily=config.FONT_FAMILY)
            axes[1, 1].legend()
        
        plt.tight_layout()
        return fig 