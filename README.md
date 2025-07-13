# 🧠 MRI Tumor Classification Pipeline

A comprehensive machine learning pipeline for classifying brain MRI scans as either containing a tumor or not, using Vision Transformers (ViT) and a modern Streamlit web interface.

## 🚀 Features

- **Vision Transformer Model**: State-of-the-art image classification using Google's ViT architecture
- **Interactive Web App**: Beautiful Streamlit interface with Tw Cen MT font styling
- **Real-time Prediction**: Upload MRI images and get instant classification results
- **Comprehensive Analysis**: Detailed statistics, visualizations, and confidence scores
- **Modular Architecture**: Clean, maintainable code structure
- **Local Dataset Support**: Works with your local MRI dataset

## 📁 Project Structure

```
Explainable AI for Computer Vision/
├── mri_data_full/                    # MRI dataset
│   ├── train/
│   │   ├── tumor/                    # Training images with tumors
│   │   └── notumor/                  # Training images without tumors
│   └── test/
│       ├── tumor/                    # Test images with tumors
│       └── notumor/                  # Test images without tumors
├── mri_pipeline_package/             # Core ML pipeline
│   ├── config.py                     # Configuration settings
│   ├── data_loader.py                # Dataset loading and preprocessing
│   ├── model_manager.py              # Model training and inference
│   ├── visualization_utils.py        # Visualization tools
│   ├── __init__.py
│   ├── model.py                      # Original model file (legacy)
│   ├── preprocessing.py              # Original preprocessing (legacy)
│   ├── utils.py                      # Original utilities (legacy)
│   ├── phantom.py                    # MRI phantom simulation
│   └── visualization.py              # Original visualization (legacy)
├── streamlit_app.py                  # Main Streamlit web application
├── train_model.py                    # Standalone training script
├── test_dataset_loading.py           # Dataset testing script
├── requirements_streamlit.txt        # Dependencies for Streamlit app
├── requirements.txt                  # Basic dependencies
└── README.md                         # This file
```

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd "Explainable AI for Computer Vision"
```

### 2. Install Dependencies

For the Streamlit app (recommended):
```bash
pip install -r requirements_streamlit.txt
```

For basic functionality:
```bash
pip install -r requirements.txt
```

### 3. Verify Dataset
Ensure your MRI dataset is in the correct structure:
```
mri_data_full/
├── train/
│   ├── tumor/        # Training images with tumors
│   └── notumor/      # Training images without tumors
└── test/
    ├── tumor/        # Test images with tumors
    └── notumor/      # Test images without tumors
```

## 🚀 Usage

### Option 1: Streamlit Web App (Recommended)

1. **Start the Streamlit app**:
```bash
streamlit run streamlit_app.py
```

2. **Open your browser** and navigate to the provided URL (usually `http://localhost:8501`)

3. **Use the app**:
   - Click "Load Dataset" in the sidebar to load your MRI data
   - Choose to either "Load Existing Model" or "Train New Model"
   - Upload an MRI image using the file uploader
   - Click "Analyze Image" to get classification results

### Option 2: Command Line Training

1. **Train the model**:
```bash
python train_model.py
```

2. **Test dataset loading**:
```bash
python test_dataset_loading.py
```

## 🎨 Web App Features

### 📊 Dataset Overview
- **Statistics Dashboard**: View training/test set sizes and class distribution
- **Sample Images**: Browse through sample MRI images from both classes
- **Interactive Plots**: Explore data distribution and characteristics

### 🔍 Image Analysis
- **Real-time Upload**: Drag and drop or browse for MRI images
- **Instant Classification**: Get tumor/no-tumor predictions with confidence scores
- **Detailed Analysis**: View image statistics, histograms, and probability distributions
- **Visual Results**: Color-coded predictions with probability charts

### 🎛️ Control Panel
- **Dataset Management**: Load and verify your MRI dataset
- **Model Management**: Train new models or load existing ones
- **Status Monitoring**: Real-time status indicators for dataset and model

## 🔧 Configuration

Edit `mri_pipeline_package/config.py` to customize:

- **Model Settings**: Change model checkpoint, image size, batch size
- **Training Parameters**: Adjust learning rate, epochs, weight decay
- **Visualization**: Modify colors, fonts, figure sizes
- **Paths**: Update dataset and model paths

## 📈 Model Performance

The Vision Transformer model typically achieves:
- **Accuracy**: 85-95% on test set
- **F1 Score**: 0.85-0.95
- **Precision**: 0.85-0.95
- **Recall**: 0.85-0.95

*Note: Performance may vary depending on dataset quality and training parameters*

## 🎯 Key Improvements Made

### ✅ Fixed Issues
- **Removed Colab-specific code**: Eliminated Google Drive mounting and shell commands
- **Fixed undefined variables**: Resolved all variable reference errors
- **Standardized framework**: Consistent PyTorch usage throughout
- **Modular architecture**: Clean separation of concerns
- **Error handling**: Comprehensive exception handling

### 🆕 New Features
- **Streamlit web interface**: Modern, interactive web app
- **Tw Cen MT font styling**: Professional appearance
- **Real-time predictions**: Instant image classification
- **Comprehensive visualizations**: Rich data analysis tools
- **Local dataset support**: Works with your existing MRI data

## 🔍 Technical Details

### Model Architecture
- **Base Model**: Google ViT-Base-Patch16-224-In21k
- **Input Size**: 224x224 pixels
- **Output**: Binary classification (tumor/no-tumor)
- **Framework**: PyTorch with Hugging Face Transformers

### Data Preprocessing
- **Resize**: All images resized to 224x224
- **Normalization**: ImageNet mean/std normalization
- **Augmentation**: Random crop, flip, rotation (training only)

### Training Configuration
- **Batch Size**: 16
- **Epochs**: 15
- **Learning Rate**: 3e-5
- **Optimizer**: AdamW with weight decay
- **Loss**: Cross-entropy

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   pip install --upgrade -r requirements_streamlit.txt
   ```

2. **CUDA Issues**:
   - The model automatically uses CPU if CUDA is not available
   - For GPU training, ensure PyTorch is installed with CUDA support

3. **Dataset Loading Errors**:
   - Verify dataset structure matches the expected format
   - Check file permissions and paths
   - Run `python test_dataset_loading.py` to diagnose issues

4. **Memory Issues**:
   - Reduce batch size in `config.py`
   - Use smaller image sizes
   - Close other applications to free memory

### Getting Help

1. **Check the logs**: Look for error messages in the terminal
2. **Verify dependencies**: Ensure all packages are installed correctly
3. **Test components**: Use the test scripts to isolate issues
4. **Check dataset**: Verify your MRI data is in the correct format

## 📝 License

This project is for educational and research purposes. Please ensure you have appropriate permissions for any medical data used.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📚 References

- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Note**: This application is for educational and research purposes. For medical diagnosis, always consult with qualified healthcare professionals. 