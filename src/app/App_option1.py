import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Rice Yield Prediction System",
    layout="wide",
    page_icon="🌾",
    initial_sidebar_state="expanded",
)

# ==========================================
# CUSTOM CSS (PROFESSIONAL THEME)
# ==========================================
st.markdown(
    """
    <style>
    :root {
        --primary: #0f766e;
        --primary-light: #14b8a6;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --border: #334155;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
    }
    
    body {
        background-color: var(--bg-dark);
        color: var(--text-primary);
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
        padding: 40px 20px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(15, 118, 110, 0.3);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5em;
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        margin: 10px 0 0 0;
        font-size: 1.1em;
    }
    
    .card {
        background: var(--bg-card);
        padding: 25px;
        border-radius: 12px;
        border: 1px solid var(--border);
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .card h2 {
        color: var(--primary-light);
        margin-top: 0;
        font-size: 1.3em;
        margin-bottom: 20px;
    }
    
    .result-box {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
        padding: 35px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(15, 118, 110, 0.3);
        margin: 25px 0;
    }
    
    .result-box .label {
        color: rgba(255, 255, 255, 0.85);
        font-size: 0.95em;
        margin-bottom: 12px;
    }
    
    .result-box .value {
        color: white;
        font-size: 2.5em;
        font-weight: 700;
        margin: 10px 0;
    }
    
    .result-box .unit {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9em;
    }
    
    .input-section {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
    }
    
    .feature-group {
        background: #0f172a;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid var(--border);
    }
    
    .status-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        margin-right: 10px;
    }
    
    .status-success {
        background-color: rgba(15, 118, 110, 0.2);
        color: #14b8a6;
    }
    
    .status-warning {
        background-color: rgba(217, 119, 6, 0.2);
        color: #fbbf24;
    }
    
    .button-container {
        display: flex;
        gap: 10px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# TITLE & HEADER
# ==========================================
st.markdown(
    """
    <div class='main-header'>
        <h1>🌾 Rice Yield Prediction System</h1>
        <p>Advanced ML-powered prediction for optimal agricultural planning</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### 📊 System Information")
    st.info(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.markdown("### ⚙️ About")
    st.markdown("""
    This prediction system uses a trained machine learning model to estimate rice yield 
    based on agricultural and environmental factors.
    
    **Key Features:**
    - Real-time predictions
    - Feature scaling & normalization
    - Box-Cox transformation support
    - Input validation
    """)

# ==========================================
# SAFE LOADER WITH VALIDATION
# ==========================================
def safe_load(path, description):
    """Safely load pickled files with error handling"""
    if not os.path.exists(path):
        st.error(f"❌ **Missing file:** `{path}`\n\nCannot load {description}. "
                 "Please ensure all required model files are in the application directory.")
        st.stop()
    try:
        return pickle.load(open(path, "rb"))
    except Exception as e:
        st.error(f"❌ **Error loading {description}:** {str(e)}")
        st.stop()


@st.cache_resource
def load_model():
    """Load and validate all model components"""
    model = safe_load("model.pkl", "Trained Model")
    scaler = safe_load("scaler.pkl", "Feature Scaler")
    
    # Load Box-Cox lambda
    try:
        lam = safe_load("lambda.pkl", "Box-Cox Lambda")
        if isinstance(lam, (list, dict)):
            st.warning("⚠️ Box-Cox lambda contains invalid data. Using raw predictions.")
            lam = None
    except:
        lam = None
    
    selected_features = safe_load("selected_features.pkl", "Selected Features")
    
    # Validate feature count
    if len(selected_features) != scaler.scale_.shape[0]:
        st.error(f"❌ **Configuration Error:** Model expects {scaler.scale_.shape[0]} features, "
                f"but {len(selected_features)} features found.")
        st.stop()
    
    return model, scaler, lam, selected_features


# Load model components
model, scaler, lam, selected_features = load_model()

# ==========================================
# INVERSE BOX-COX TRANSFORMATION
# ==========================================
def inverse_boxcox(y_bc, lam):
    """Safely reverse Box-Cox transformation"""
    if lam is None:
        return y_bc
    try:
        if abs(lam) < 1e-10:  # lam ≈ 0
            return np.exp(y_bc)
        else:
            transformed = lam * y_bc + 1
            if np.any(transformed <= 0):
                return y_bc
            return np.power(transformed, 1 / lam)
    except Exception as e:
        st.warning(f"⚠️ Box-Cox inversion failed: {str(e)}")
        return y_bc


# ==========================================
# INPUT SECTION
# ==========================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 📝 Input Features")

input_data = {}
cols = st.columns(2)

for i, feature in enumerate(selected_features):
    with cols[i % 2]:
        input_data[feature] = st.number_input(
            label=feature,
            value=0.0,
            format="%.5f",
            help=f"Enter value for {feature}"
        )

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# PREDICTION LOGIC
# ==========================================
col1, col2 = st.columns([3, 1])

with col1:
    predict_btn = st.button("🚀 Predict Rice Yield", use_container_width=True, type="primary")

with col2:
    reset_btn = st.button("↻ Reset", use_container_width=True)

if reset_btn:
    st.rerun()

if predict_btn:
    try:
        # Convert to DataFrame
        X_input = pd.DataFrame([input_data])
        
        # Validate inputs
        if (X_input < 0).any().any():
            st.warning("⚠️ Some features contain negative values. Ensure all inputs are non-negative.")
        
        # Scale features
        X_scaled = scaler.transform(X_input)
        
        # Make prediction
        pred_transformed = model.predict(X_scaled)[0]
        
        # Reverse transformations
        pred_final = inverse_boxcox(pred_transformed, lam)
        
        # Ensure positive result
        if pred_final < 0:
            pred_final = 0
        
        # Display result
        st.markdown(
            f"""
            <div class='result-box'>
                <div class='label'>Predicted Rice Yield</div>
                <div class='value'>{pred_final:,.2f}</div>
                <div class='unit'>Kilograms per Hectare (Kg/Ha)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", "✅ Success", delta=None)
        with col2:
            st.metric("Model Features", len(selected_features), delta=None)
        with col3:
            st.metric("Box-Cox Applied", "Yes" if lam else "No", delta=None)
        
    except Exception as e:
        st.error(f"❌ **Prediction Failed:** {str(e)}\n\nPlease verify your inputs and try again.")

# ==========================================
# MODEL DETAILS SECTION
# ==========================================
with st.expander("📋 Model Details & Configuration", expanded=False):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Features Used in Model:**")
        feature_df = pd.DataFrame({"Feature": selected_features, "Index": range(1, len(selected_features) + 1)})
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("**Model Configuration:**")
        config_data = {
            "Parameter": ["Total Features", "Box-Cox Lambda", "Model Status"],
            "Value": [len(selected_features), f"{lam:.6f}" if lam else "None", "✅ Loaded"]
        }
        st.dataframe(pd.DataFrame(config_data), use_container_width=True, hide_index=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: var(--text-secondary); font-size: 0.9em; margin-top: 20px;'>
    <p>🌾 Rice Yield Prediction System | Agricultural ML Platform | v1.0</p>
    <p>For support or issues, please contact the system administrator.</p>
    </div>
    """,
    unsafe_allow_html=True
)