"""
Model management for MRI tumor classification using Vision Transformers
"""

import torch
import torch.nn as nn
from transformers import (
    ViTForImageClassification, 
    AutoImageProcessor,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import config

class MRIModelManager:
    """Manages Vision Transformer model for MRI classification"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.model = None
        self.image_processor = None
        self.trainer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def setup_model(self):
        """Setup the Vision Transformer model"""
        try:
            # Load image processor
            self.image_processor = AutoImageProcessor.from_pretrained(config.MODEL_CHECKPOINT)
            
            # Setup label mappings
            id2label = {str(i): label for i, label in enumerate(config.LABELS)}
            label2id = {label: str(i) for i, label in enumerate(config.LABELS)}
            
            # Load pre-trained model
            self.model = ViTForImageClassification.from_pretrained(
                config.MODEL_CHECKPOINT,
                num_labels=config.NUM_CLASSES,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True
            )
            
            # Move model to device
            self.model.to(self.device)
            
            print(f"✅ Model setup completed on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up model: {e}")
            return False
    
    def setup_trainer(self, output_dir="./results"):
        """Setup the trainer with training arguments"""
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=config.NUM_EPOCHS,
                per_device_train_batch_size=config.BATCH_SIZE,
                per_device_eval_batch_size=config.BATCH_SIZE,
                warmup_steps=config.WARMUP_STEPS,
                weight_decay=config.WEIGHT_DECAY,
                logging_dir="./logs",
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                push_to_hub=False,
                report_to=None
            )
            
            # Setup trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.data_loader.dataset["train"],
                eval_dataset=self.data_loader.dataset["test"],
                tokenizer=self.image_processor,
                data_collator=self.data_loader.data_collator,
                compute_metrics=self._compute_metrics
            )
            
            print("✅ Trainer setup completed!")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up trainer: {e}")
            return False
    
    def train_model(self):
        """Train the model"""
        try:
            if self.trainer is None:
                print("❌ Trainer not setup. Call setup_trainer() first.")
                return False
                
            print("🚀 Starting model training...")
            train_result = self.trainer.train()
            
            # Save the model
            self.trainer.save_model()
            
            print("✅ Training completed!")
            print(f"📊 Training loss: {train_result.training_loss:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during training: {e}")
            return False
    
    def evaluate_model(self):
        """Evaluate the model on test set"""
        try:
            if self.trainer is None:
                print("❌ Trainer not setup. Call setup_trainer() first.")
                return None
                
            print("🔍 Evaluating model...")
            eval_result = self.trainer.evaluate()
            
            print("✅ Evaluation completed!")
            print(f"📊 Test accuracy: {eval_result['eval_accuracy']:.4f}")
            print(f"📊 Test loss: {eval_result['eval_loss']:.4f}")
            
            return eval_result
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            return None
    
    def predict_single_image(self, image):
        """Predict class for a single image"""
        try:
            if self.model is None:
                print("❌ Model not setup. Call setup_model() first.")
                return None
                
            # Preprocess image
            inputs = self.image_processor(image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class = torch.argmax(logits, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Get label
            predicted_label = config.LABELS[predicted_class]
            
            return {
                'predicted_class': predicted_class,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'probabilities': probabilities[0].cpu().numpy()
            }
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None
    
    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        acc = accuracy_score(labels, predictions)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def save_model(self, path="./saved_model"):
        """Save the trained model"""
        try:
            if self.model is None:
                print("❌ Model not setup.")
                return False
                
            self.model.save_pretrained(path)
            self.image_processor.save_pretrained(path)
            print(f"✅ Model saved to {path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, path="./saved_model"):
        """Load a trained model"""
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(path)
            self.model = ViTForImageClassification.from_pretrained(path)
            self.model.to(self.device)
            print(f"✅ Model loaded from {path}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False 