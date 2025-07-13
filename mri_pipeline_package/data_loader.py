"""
Data loading and preprocessing for MRI tumor classification
"""

import os
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from transformers import AutoImageProcessor, DefaultDataCollator
import config

class MRIDataLoader:
    """Handles loading and preprocessing of MRI dataset"""
    
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or config.DATASET_PATH
        self.image_processor = None
        self.data_collator = None
        self.dataset = None
        self.label2id = {}
        self.id2label = {}
        
    def load_dataset(self):
        """Load the MRI dataset using Hugging Face datasets"""
        try:
            self.dataset = load_dataset(
                path="imagefolder",
                data_dir=self.data_dir,
                split={
                    "train": "train",
                    "test": "test"
                }
            )
            
            # Extract label information
            labels = self.dataset["train"].features["label"].names
            for i, label in enumerate(labels):
                self.label2id[label] = str(i)
                self.id2label[str(i)] = label
                
            print(f"✅ Dataset loaded successfully!")
            print(f"📊 Train samples: {len(self.dataset['train'])}")
            print(f"📊 Test samples: {len(self.dataset['test'])}")
            print(f"🏷️ Labels: {labels}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def setup_preprocessing(self):
        """Setup image processor and data collator"""
        try:
            # Load image processor
            self.image_processor = AutoImageProcessor.from_pretrained(config.MODEL_CHECKPOINT)
            
            # Setup data collator
            self.data_collator = DefaultDataCollator(return_tensors="pt")
            
            # Define transforms
            self.transforms = self._get_transforms()
            
            # Apply transforms to dataset
            self.dataset["train"].set_transform(self._preprocess_train)
            self.dataset["test"].set_transform(self._preprocess_val)
            
            print("✅ Preprocessing setup completed!")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up preprocessing: {e}")
            return False
    
    def _get_transforms(self):
        """Get image transformation pipeline"""
        normalize = transforms.Normalize(
            mean=self.image_processor.image_mean, 
            std=self.image_processor.image_std
        )
        
        size = (
            self.image_processor.size["shortest_edge"]
            if "shortest_edge" in self.image_processor.size
            else (self.image_processor.size["height"], self.image_processor.size["width"])
        )
        
        return transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.ToTensor(),
            normalize
        ])
    
    def _preprocess_train(self, example_batch):
        """Preprocess training examples with augmentation"""
        images = [
            self.transforms(image.convert("RGB")) for image in example_batch["image"]
        ]
        example_batch["pixel_values"] = images
        return example_batch
    
    def _preprocess_val(self, example_batch):
        """Preprocess validation examples without augmentation"""
        images = [
            self.transforms(image.convert("RGB")) for image in example_batch["image"]
        ]
        example_batch["pixel_values"] = images
        return example_batch
    
    def get_sample_images(self, split="train", num_samples=9):
        """Get sample images for visualization"""
        if self.dataset is None:
            return None
            
        samples = []
        for i in range(min(num_samples, len(self.dataset[split]))):
            sample = self.dataset[split][i]
            samples.append({
                'image': sample['image'],
                'label': sample['label'],
                'label_name': self.id2label[str(sample['label'])]
            })
        return samples
    
    def get_class_distribution(self):
        """Get class distribution statistics"""
        if self.dataset is None:
            return None
            
        train_labels = [sample['label'] for sample in self.dataset['train']]
        test_labels = [sample['label'] for sample in self.dataset['test']]
        
        train_dist = np.bincount(train_labels)
        test_dist = np.bincount(test_labels)
        
        return {
            'train': {self.id2label[str(i)]: count for i, count in enumerate(train_dist)},
            'test': {self.id2label[str(i)]: count for i, count in enumerate(test_dist)}
        } 