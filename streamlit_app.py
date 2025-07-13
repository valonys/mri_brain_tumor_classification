"""
Streamlit Web Application for MRI Tumor Classification (Cloud Optimized)
"""

import streamlit as st
import os
import sys
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the mri_pipeline_package to the path
sys.path.append('./mri_pipeline_package')

# Import our custom modules with error handling
try:
    from mri_pipeline_package.config import *
    from mri_pipeline_package.data_loader import MRIDataLoader
    from mri_pipeline_package.model_manager import MRIModelManager
    from mri_pipeline_package.visualization_utils import MRIVisualizer
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.info("Please ensure all dependencies are installed correctly.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🧠 MRI Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Tw Cen MT font
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tw+Cen+MT:wght@400;500;600;700&display=swap');
    
    .main-header {
        font-family: 'Tw Cen MT', sans-serif !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1f77b4 !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
    }
    
    .sub-header {
        font-family: 'Tw Cen MT', sans-serif !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #2c3e50 !important;
        margin-bottom: 1rem !important;
    }
    
    .description {
        font-family: 'Tw Cen MT', sans-serif !important;
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
        color: #34495e !important;
        text-align: justify !important;
        margin-bottom: 2rem !important;
    }
    
    .metric-card {
        font-family: 'Tw Cen MT', sans-serif !important;
        background-color: #f8f9fa !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        border-left: 4px solid #1f77b4 !important;
        margin: 0.5rem 0 !important;
    }
    
    .prediction-result {
        font-family: 'Tw Cen MT', sans-serif !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        margin: 1rem 0 !important;
    }
    
    .tumor-prediction {
        background-color: #ffebee !important;
        border-left: 4px solid #f44336 !important;
        color: #c62828 !important;
    }
    
    .notumor-prediction {
        background-color: #e8f5e8 !important;
        border-left: 4px solid #4caf50 !important;
        color: #2e7d32 !important;
    }
    
    .stButton > button {
        font-family: 'Tw Cen MT', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 25px !important;
        padding: 0.5rem 2rem !important;
    }
    
    .stSelectbox > div > div {
        font-family: 'Tw Cen MT', sans-serif !important;
    }
    
    .stFileUploader > div > div {
        font-family: 'Tw Cen MT', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = None
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

def initialize_components():
    """Initialize the main components"""
    try:
        if st.session_state.data_loader is None:
            st.session_state.data_loader = MRIDataLoader()
        if st.session_state.visualizer is None:
            st.session_state.visualizer = MRIVisualizer()
    except Exception as e:
        st.error(f"❌ Error initializing components: {e}")
        return False
    return True

def load_dataset():
    """Load the MRI dataset"""
    try:
        with st.spinner("Loading MRI dataset..."):
            success = st.session_state.data_loader.load_dataset()
            if success:
                st.session_state.data_loader.setup_preprocessing()
                st.success("✅ Dataset loaded successfully!")
                return True
            else:
                st.error("❌ Failed to load dataset!")
                return False
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return False

def load_or_train_model():
    """Load existing model or train a new one"""
    model_path = "./saved_model"
    
    if os.path.exists(model_path) and st.button("Load Existing Model"):
        try:
            with st.spinner("Loading existing model..."):
                if st.session_state.model_manager is None:
                    st.session_state.model_manager = MRIModelManager(st.session_state.data_loader)
                
                success = st.session_state.model_manager.load_model(model_path)
                if success:
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                    return True
                else:
                    st.error("❌ Failed to load model!")
                    return False
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return False
    
    elif st.button("Train New Model"):
        try:
            with st.spinner("Setting up and training model..."):
                if st.session_state.model_manager is None:
                    st.session_state.model_manager = MRIModelManager(st.session_state.data_loader)
                
                # Setup model
                if not st.session_state.model_manager.setup_model():
                    st.error("❌ Failed to setup model!")
                    return False
                
                # Setup trainer
                if not st.session_state.model_manager.setup_trainer():
                    st.error("❌ Failed to setup trainer!")
                    return False
                
                # Train model
                if st.session_state.model_manager.train_model():
                    st.session_state.model_loaded = True
                    st.success("✅ Model trained successfully!")
                    return True
                else:
                    st.error("❌ Failed to train model!")
                    return False
        except Exception as e:
            st.error(f"❌ Error training model: {e}")
            return False
    
    return False

def predict_image(image):
    """Predict tumor classification for uploaded image"""
    if not st.session_state.model_loaded:
        st.error("❌ Model not loaded. Please load or train a model first.")
        return None
    
    try:
        with st.spinner("Analyzing image..."):
            prediction = st.session_state.model_manager.predict_single_image(image)
            return prediction
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        return None

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 MRI Tumor Classification</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">This application uses a Vision Transformer (ViT) to classify MRI brain scans as either containing a tumor or not. Upload your MRI image to get instant classification results.</p>', unsafe_allow_html=True)
    
    # Initialize components
    if not initialize_components():
        st.stop()
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # Dataset loading
    st.sidebar.markdown("### 📊 Dataset")
    if st.sidebar.button("Load Dataset"):
        load_dataset()
    
    # Model management
    st.sidebar.markdown("### 🤖 Model")
    if st.session_state.data_loader and st.session_state.data_loader.dataset is not None:
        load_or_train_model()
    
    # File upload
    st.sidebar.markdown("### 📁 Upload MRI Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an MRI image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a brain MRI image for tumor classification"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
        
        if st.session_state.data_loader and st.session_state.data_loader.dataset is not None:
            # Dataset statistics
            distribution = st.session_state.data_loader.get_class_distribution()
            
            if distribution:
                col1_1, col1_2 = st.columns(2)
                
                with col1_1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Total Training Images", sum(distribution['train'].values()))
                    st.metric("Total Test Images", sum(distribution['test'].values()))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col1_2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    for label, count in distribution['train'].items():
                        st.metric(f"Training {label.title()}", count)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Class distribution plot
                try:
                    fig = st.session_state.visualizer.plot_class_distribution(distribution)
                    if fig:
                        st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not display distribution plot: {e}")
                
                # Sample images
                st.markdown('<h3 class="sub-header">📸 Sample Images</h3>', unsafe_allow_html=True)
                
                # Get sample images
                try:
                    train_samples = st.session_state.data_loader.get_sample_images("train", 9)
                    if train_samples:
                        fig = st.session_state.visualizer.plot_sample_images(train_samples, "Training Samples")
                        if fig:
                            st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not display sample images: {e}")
        else:
            st.info("👆 Please load the dataset using the sidebar button.")
    
    with col2:
        st.markdown('<h2 class="sub-header">🔍 Image Analysis</h2>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Display uploaded image
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded MRI Image", use_column_width=True)
                
                # Make prediction
                if st.button("🔬 Analyze Image", type="primary"):
                    prediction = predict_image(image)
                    
                    if prediction:
                        # Display prediction results
                        label = prediction['predicted_label']
                        confidence = prediction['confidence']
                        probabilities = prediction['probabilities']
                        
                        # Styled prediction result
                        css_class = "tumor-prediction" if label == "tumor" else "notumor-prediction"
                        st.markdown(
                            f'<div class="prediction-result {css_class}">'
                            f'<strong>Prediction:</strong> {label.title()}<br>'
                            f'<strong>Confidence:</strong> {confidence:.2%}'
                            '</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Probability chart
                        try:
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=["No Tumor", "Tumor"],
                                    y=probabilities,
                                    marker_color=['#2E8B57', '#DC143C'],
                                    text=[f'{p:.3f}' for p in probabilities],
                                    textposition='auto'
                                )
                            ])
                            
                            fig.update_layout(
                                title="Classification Probabilities",
                                xaxis_title="Class",
                                yaxis_title="Probability",
                                font=dict(family="Tw Cen MT", size=14),
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not display probability chart: {e}")
                        
                        # Detailed analysis
                        st.markdown('<h3 class="sub-header">📈 Detailed Analysis</h3>', unsafe_allow_html=True)
                        
                        # Image statistics
                        try:
                            img_array = np.array(image)
                            if len(img_array.shape) == 3:
                                img_array = np.mean(img_array, axis=2)
                            
                            col2_1, col2_2, col2_3 = st.columns(3)
                            with col2_1:
                                st.metric("Mean Intensity", f"{np.mean(img_array):.2f}")
                            with col2_2:
                                st.metric("Std Intensity", f"{np.std(img_array):.2f}")
                            with col2_3:
                                st.metric("Max Intensity", f"{np.max(img_array):.0f}")
                            
                            # Histogram
                            fig = px.histogram(
                                x=img_array.ravel(),
                                nbins=50,
                                title="Image Intensity Distribution",
                                labels={'x': 'Pixel Intensity', 'y': 'Frequency'}
                            )
                            fig.update_layout(font=dict(family="Tw Cen MT", size=14))
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not display detailed analysis: {e}")
            except Exception as e:
                st.error(f"❌ Error processing uploaded image: {e}")
        else:
            st.info("👆 Please upload an MRI image using the sidebar.")
    
    # Model status
    st.sidebar.markdown("### 📋 Status")
    if st.session_state.data_loader and st.session_state.data_loader.dataset is not None:
        st.sidebar.success("✅ Dataset Loaded")
    else:
        st.sidebar.error("❌ Dataset Not Loaded")
    
    if st.session_state.model_loaded:
        st.sidebar.success("✅ Model Ready")
    else:
        st.sidebar.warning("⚠️ Model Not Ready")

if __name__ == "__main__":
    main() 