"""
EcoSort AI - Streamlit Web Application
Interactive E-Waste Detection System
"""

import streamlit as st
import numpy as np
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="EcoSort AI - E-Waste Detector",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .ewaste-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .safe-alert {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #2E7D32;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #1B5E20;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
IMG_SIZE = 224
MODEL_PATH = 'ecosort_best_model.h5'

# E-waste information
EWASTE_INFO = {
    'dangers': [
        "🔴 Contains toxic heavy metals like lead, mercury, and cadmium",
        "🔴 Can contaminate soil and groundwater if improperly disposed",
        "🔴 Releases harmful chemicals when burned or broken down",
        "🔴 May contain valuable materials that should be recycled",
        "🔴 Poses health risks to humans and wildlife"
    ],
    'disposal': [
        "✅ Take to certified e-waste recycling centers",
        "✅ Check with manufacturers for take-back programs",
        "✅ Never throw in regular trash bins",
        "✅ Remove batteries before disposal if possible",
        "✅ Donate working electronics instead of discarding"
    ],
    'materials': [
        "💎 Precious metals: Gold, silver, copper",
        "🔧 Plastics and circuit boards",
        "⚡ Rare earth elements",
        "🪟 Glass and ceramics",
        "⚠️ Hazardous materials requiring special handling"
    ],
    'examples': {
        'ewaste': [
            "📱 Mobile phones and tablets",
            "💻 Computers and laptops",
            "🖥️ Monitors and TVs",
            "⌨️ Keyboards and mice",
            "🔋 Batteries and chargers",
            "🖨️ Printers and scanners",
            "🎮 Gaming consoles",
            "📷 Cameras and accessories"
        ],
        'non_ewaste': [
            "🗑️ Organic waste",
            "📦 Paper and cardboard",
            "🥤 Plastic bottles",
            "🍾 Glass bottles",
            "🥫 Metal cans",
            "👕 Textiles",
            "🪵 Wood",
            "🍽️ Food waste"
        ]
    }
}

@st.cache_resource
def load_model():
    """Load model with proper architecture reconstruction"""
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        import h5py
        
        # First, try to load the weights file
        with h5py.File(MODEL_PATH, 'r') as f:
            # Check if file is valid
            if 'model_weights' not in f.keys() and 'model_config' not in f.keys():
                st.info("Creating model architecture from scratch...")
        
        # Rebuild the exact architecture
        base_model = keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        # Build model
        inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Try to load weights
        try:
            model.load_weights(MODEL_PATH)
            st.success("✅ Model loaded successfully with weights!")
        except:
            st.warning("⚠️ Using model with ImageNet weights only (no custom training)")
        
        return model, None
        
    except Exception as e:
        return None, str(e)

def preprocess_image(image):
    """Preprocess image for model"""
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_image(model, image):
    """Classify image"""
    img_array = preprocess_image(image)
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    is_ewaste = prediction > 0.5
    confidence = prediction if is_ewaste else (1 - prediction)
    
    return {
        'is_ewaste': bool(is_ewaste),
        'confidence': float(confidence),
        'raw_prediction': float(prediction)
    }

def display_confidence_meter(confidence):
    """Display confidence meter"""
    st.markdown("### Confidence Level")
    
    if confidence > 0.8:
        st.success(f"**High Confidence: {confidence*100:.1f}%**")
    elif confidence > 0.6:
        st.warning(f"**Medium Confidence: {confidence*100:.1f}%**")
    else:
        st.error(f"**Low Confidence: {confidence*100:.1f}%**")
    
    st.progress(confidence)

def main():
    """Main app"""
    
    # Header
    st.markdown('<h1 class="main-header">♻️ EcoSort AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered E-Waste Detection System</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### About EcoSort AI")
        st.info(
            "EcoSort AI uses advanced computer vision to identify electronic waste "
            "and provide proper disposal guidance. Help protect our environment! 🌍"
        )
        
        st.markdown("### 📊 Statistics")
        st.metric("E-waste Generated Globally", "57.4M tons/year")
        st.metric("Properly Recycled", "17.4%")
        st.metric("Recovery Rate Possible", "95%")
        
        st.markdown("### 🎯 Model Info")
        st.write("- **Architecture:** MobileNetV2")
        st.write("- **Input Size:** 224x224")
        st.write("- **Classes:** Binary (E-waste/Not)")
        
        st.markdown("---")
        st.markdown("### 👨‍💻 Developer")
        st.write("Created with ❤️ by Avrelya")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of waste to classify"
        )
        
        st.markdown("#### Or use your camera:")
        camera_image = st.camera_input("Take a picture")
        
        image_source = uploaded_file if uploaded_file else camera_image
        
        if image_source:
            image = Image.open(image_source).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Analyze Image", type="primary"):
                model, error = load_model()
                
                if error:
                    st.error(f"❌ Error loading model: {error}")
                    return
                
                with st.spinner('Analyzing image...'):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    result = classify_image(model, image)
                    st.session_state['result'] = result
                    st.rerun()
    
    with col2:
        st.markdown("### 📊 Results")
        
        if 'result' in st.session_state:
            result = st.session_state['result']
            
            if result['is_ewaste']:
                st.markdown(
                    '<div class="ewaste-alert">'
                    '<h2>⚠️ E-WASTE DETECTED</h2>'
                    '<p>This item is classified as electronic waste and requires special handling.</p>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="safe-alert">'
                    '<h2>✅ NOT E-WASTE</h2>'
                    '<p>This item is not classified as electronic waste.</p>'
                    '</div>',
                    unsafe_allow_html=True
                )
            
            display_confidence_meter(result['confidence'])
            
            if result['is_ewaste']:
                st.markdown("---")
                
                tab1, tab2, tab3 = st.tabs(["⚠️ Dangers", "♻️ Disposal", "💎 Materials"])
                
                with tab1:
                    st.markdown("### Environmental & Health Dangers")
                    for danger in EWASTE_INFO['dangers']:
                        st.markdown(f"- {danger}")
                
                with tab2:
                    st.markdown("### Proper Disposal Methods")
                    for method in EWASTE_INFO['disposal']:
                        st.markdown(f"- {method}")
                    
                    st.warning(
                        "⚠️ **Important:** E-waste recycling can recover up to 95% of materials "
                        "and prevent environmental contamination. Always use certified recycling facilities."
                    )
                
                with tab3:
                    st.markdown("### Recyclable Materials")
                    for material in EWASTE_INFO['materials']:
                        st.markdown(f"- {material}")
        else:
            st.info("👆 Upload an image to get started!")
            
            st.markdown("### 📋 What is E-Waste?")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**E-Waste Examples:**")
                for item in EWASTE_INFO['examples']['ewaste']:
                    st.markdown(f"- {item}")
            
            with col_b:
                st.markdown("**Non E-Waste Examples:**")
                for item in EWASTE_INFO['examples']['non_ewaste']:
                    st.markdown(f"- {item}")
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Made with ❤️ for a sustainable future | EcoSort AI © 2024"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
