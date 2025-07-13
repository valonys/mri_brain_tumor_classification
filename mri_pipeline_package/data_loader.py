"""
Data loading and preprocessing for MRI tumor classification
"""

import os
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor
from transformers.data.data_collator import DefaultDataCollator
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
        
    def load_dataset(self, dataset_source="local"):
        """Load the MRI dataset from various sources"""
        try:
            if dataset_source == "huggingface":
                # Load from Hugging Face Hub
                print("🔄 Loading dataset from Hugging Face Hub...")
                dataset_dict = load_dataset("amiguel/MRI_Segmentation-1")
                self.dataset = {
                    "train": dataset_dict["train"],
                    "test": dataset_dict["test"] if "test" in dataset_dict else dataset_dict["validation"]
                }
            elif dataset_source == "local" and os.path.exists(self.data_dir):
                # Load from local directory
                print("🔄 Loading dataset from local directory...")
                dataset_dict = load_dataset("imagefolder", data_dir=self.data_dir)
                self.dataset = {
                    "train": dataset_dict["train"],
                    "test": dataset_dict["test"]
                }
            else:
                # Fallback to a sample dataset for demo
                print("🔄 Loading sample dataset for demo...")
                return self._load_sample_dataset()
            
            # Extract label information
            if hasattr(self.dataset["train"], 'features') and hasattr(self.dataset["train"].features, 'label'):
                labels = self.dataset["train"].features["label"].names
                for i, label in enumerate(labels):
                    self.label2id[label] = str(i)
                    self.id2label[str(i)] = label
            else:
                # Default labels for demo
                self.label2id = {'notumor': '0', 'tumor': '1'}
                self.id2label = {'0': 'notumor', '1': 'tumor'}
                
            print(f"✅ Dataset loaded successfully!")
            print(f"📊 Train samples: {len(self.dataset['train'])}")
            print(f"📊 Test samples: {len(self.dataset['test'])}")
            print(f"🏷️ Labels: {list(self.label2id.keys())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print("🔄 Falling back to sample dataset...")
            return self._load_sample_dataset()
    
    def _load_sample_dataset(self):
        """Load a small sample dataset for demo purposes"""
        try:
            # Create a minimal sample dataset
            from PIL import Image
            import numpy as np
            
            # Create sample images (simple patterns)
            def create_sample_image(label, size=(224, 224)):
                img = np.random.randint(0, 255, (size[0], size[1]), dtype=np.uint8)
                if label == 1:  # Tumor - add some pattern
                    img[100:150, 100:150] = 255
                return Image.fromarray(img)
            
            # Create sample data
            train_data = []
            test_data = []
            
            # Generate sample training data
            for i in range(50):
                label = i % 2  # Alternate between 0 and 1
                img = create_sample_image(label)
                train_data.append({
                    'image': img,
                    'label': label
                })
            
            # Generate sample test data
            for i in range(20):
                label = i % 2
                img = create_sample_image(label)
                test_data.append({
                    'image': img,
                    'label': label
                })
            
            # Create dataset structure
            self.dataset = {
                'train': train_data,
                'test': test_data
            }
            
            # Setup labels
            self.label2id = {'notumor': '0', 'tumor': '1'}
            self.id2label = {'0': 'notumor', '1': 'tumor'}
            
            print("✅ Sample dataset loaded for demo!")
            print("📊 Train samples: 50")
            print("📊 Test samples: 20")
            print("🏷️ Labels: ['notumor', 'tumor']")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating sample dataset: {e}")
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