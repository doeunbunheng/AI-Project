import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

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
    
    .metric-card {
        background: #0f172a;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid var(--primary-light);
        margin: 10px 0;
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.9em;
        margin-bottom: 5px;
    }
    
    .metric-value {
        color: var(--primary-light);
        font-size: 1.6em;
        font-weight: 700;
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
        <p>Advanced ML-powered prediction with multiple preprocessing options</p>
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
    
    st.markdown("### 🔧 Choose Model Version")
    model_option = st.selectbox(
        "Select prediction model:",
        [
            "Option 1 — StandardScaler MLR",
            "Option 2 — Log-Transformed MLR",
            "Option 3 — Box-Cox MLR"
        ]
    )
    
    st.markdown("### 📋 Pipeline Steps")
    steps = [
        "✅ Load dataset",
        "✅ Data cleaning",
        "✅ Feature Extraction and EDA",
        "✅ Understand column types",
        "✅ Feature selection",
        "✅ Data preprocessing",
    ]
    for step in steps:
        st.markdown(f"- {step}")
    
    st.markdown("### ⚙️ Preprocessing Options")
    st.markdown("""
    1. **StandardScaler MLR** - Standard normalization
    2. **Log-Transformed MLR** - Log transformation
    3. **Box-Cox MLR** - Box-Cox transformation
    """)
    
    st.markdown("### ℹ️ About")
    st.markdown("""
    This prediction system uses trained machine learning models to estimate rice yield 
    based on agricultural and environmental factors.
    
    **Key Features:**
    - Real-time predictions
    - Feature scaling & normalization
    - Multiple transformation methods
    - Input validation
    - Model performance metrics
    """)

# ==========================================
# SAFE LOADER WITH VALIDATION
# ==========================================
def safe_load(path, description):
    """Safely load pickled files with error handling"""
    if not os.path.exists(path):
        st.error(f"❌ **Missing file:** `{path}`\n\nCannot load {description}.")
        return None
    try:
        return pickle.load(open(path, "rb"))
    except Exception as e:
        st.error(f"❌ **Error loading {description}:** {str(e)}")
        return None

# ==========================================
# LOAD MODELS BASED ON SELECTION
# ==========================================
@st.cache_resource
def load_selected_model(model_option):
    """Load model based on user selection"""
    
    if model_option == "Option 1 — StandardScaler MLR":
        model_files = {
            'model': 'model_option1.pkl',
            'scaler': 'scaler_option1.pkl',
            'lambda': None,
            'features': 'features_option1.pkl',
            'r2': None
        }
    elif model_option == "Option 2 — Log-Transformed MLR":
        model_files = {
            'model': 'model_option2.pkl',
            'scaler': 'scaler_option2.pkl',
            'lambda': 'log',
            'features': 'features_option2.pkl',
            'r2': None
        }
    else:  # Option 3 — Box-Cox MLR
        model_files = {
            'model': 'model_option3.pkl',
            'scaler': 'scaler_option3.pkl',
            'lambda': 'lambda_option3.pkl',
            'features': 'features_option3.pkl',
            'r2': None
        }
    
    # Load model
    model = safe_load(model_files['model'], f"{model_option} Model")
    if model is None:
        return None, None, None, None, None
    
    # Load scaler
    scaler = safe_load(model_files['scaler'], f"{model_option} Scaler")
    if scaler is None:
        return None, None, None, None, None
    
    # Load lambda if needed
    lam = None
    if model_files['lambda'] and model_files['lambda'] != 'log':
        lam = safe_load(model_files['lambda'], "Box-Cox Lambda")
    elif model_files['lambda'] == 'log':
        lam = 'log'
    
    # Load features
    features = safe_load(model_files['features'], f"{model_option} Features")
    if features is None:
        return None, None, None, None, None
    
    # Try to load R² score
    r2_score_val = None
    r2_file = f"r2_option{model_option[7]}.pkl"
    if os.path.exists(r2_file):
        r2_score_val = pickle.load(open(r2_file, 'rb'))
    
    return model, scaler, lam, features, r2_score_val

# Load selected model
model, scaler, lam, selected_features, r2_score_val = load_selected_model(model_option)

if model is None or scaler is None or selected_features is None:
    st.error("❌ Unable to load model files. Please check your model configuration.")
    st.stop()

# ==========================================
# INVERSE TRANSFORM FUNCTIONS
# ==========================================
def inverse_log(y_log):
    """Reverse log transformation"""
    return np.exp(y_log)

def inverse_boxcox(y_bc, lam):
    """Safely reverse Box-Cox transformation"""
    if lam is None or lam == 'log':
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

def inverse_transform(y, lam):
    """Apply appropriate inverse transformation"""
    if lam is None:
        return y
    elif lam == 'log':
        return inverse_log(y)
    else:
        return inverse_boxcox(y, lam)

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
# PREDICTION SECTION
# ==========================================
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    predict_btn = st.button("🚀 Predict Rice Yield", use_container_width=True, type="primary")

with col2:
    reset_btn = st.button("↻ Reset", use_container_width=True)

with col3:
    compare_btn = st.button("📊 Compare", use_container_width=True)

if reset_btn:
    st.rerun()

# ==========================================
# SINGLE PREDICTION
# ==========================================
if predict_btn:
    try:
        # Convert to DataFrame
        X_input = pd.DataFrame([input_data])
        
        # Validate inputs
        if (X_input < 0).any().any():
            st.warning("⚠️ Some features contain negative values. Results may be unreliable.")
        
        # Scale features
        X_scaled = scaler.transform(X_input)
        
        # Make prediction
        pred_transformed = model.predict(X_scaled)[0]
        
        # Reverse transformation
        pred_final = inverse_transform(pred_transformed, lam)
        
        # Ensure positive result
        if pred_final < 0:
            pred_final = 0
        
        # Display result
        st.markdown(
            f"""
            <div class='result-box'>
                <div class='label'>Predicted Rice Yield</div>
                <div class='value'>Ŷ = {pred_final:,.2f}</div>
                <div class='unit'>Kilograms per Hectare (Kg/Ha)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>Model R² Score</div>
                    <div class='metric-value'>{r2_score_val:.4f if r2_score_val else "N/A"}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>Method Used</div>
                    <div class='metric-value' style='font-size: 1.2em;'>{model_option.split(' — ')[1]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>Status</div>
                    <div class='metric-value'>✅ Success</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Display input summary
        st.markdown("**Input Features Summary:**")
        input_df = pd.DataFrame([input_data]).T
        input_df.columns = ['Value']
        st.dataframe(input_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ **Prediction Failed:** {str(e)}\n\nPlease verify your inputs and try again.")

# ==========================================
# COMPARE ALL MODELS
# ==========================================
if compare_btn:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Comparison of All Models")
    
    try:
        X_input = pd.DataFrame([input_data])
        
        # Store results for all models
        all_predictions = {}
        all_r2_scores = {}
        
        models_list = [
            ("Option 1 — StandardScaler MLR", "model_option1.pkl", "scaler_option1.pkl", None, "features_option1.pkl"),
            ("Option 2 — Log-Transformed MLR", "model_option2.pkl", "scaler_option2.pkl", "log", "features_option2.pkl"),
            ("Option 3 — Box-Cox MLR", "model_option3.pkl", "scaler_option3.pkl", "lambda_option3.pkl", "features_option3.pkl"),
        ]
        
        for name, model_file, scaler_file, lambda_type, features_file in models_list:
            try:
                # Load model
                temp_model = pickle.load(open(model_file, 'rb'))
                temp_scaler = pickle.load(open(scaler_file, 'rb'))
                
                # Load lambda
                temp_lam = None
                if lambda_type == "log":
                    temp_lam = "log"
                elif lambda_type and os.path.exists(lambda_type):
                    temp_lam = pickle.load(open(lambda_type, 'rb'))
                
                # Predict
                X_scaled = temp_scaler.transform(X_input)
                pred = temp_model.predict(X_scaled)[0]
                pred = inverse_transform(pred, temp_lam)
                pred = max(0, pred)
                
                all_predictions[name] = pred
                
                # Load R² if available
                r2_file = f"r2_option{name[7]}.pkl"
                if os.path.exists(r2_file):
                    all_r2_scores[name] = pickle.load(open(r2_file, 'rb'))
                else:
                    all_r2_scores[name] = 0.0
                    
            except Exception as e:
                st.warning(f"Could not load {name}: {str(e)}")
        
        # Create comparison table
        if all_predictions:
            comparison_df = pd.DataFrame({
                'Model': list(all_predictions.keys()),
                'Ŷ (Prediction)': [f"{v:,.2f}" for v in all_predictions.values()],
                'R² Score': [f"{all_r2_scores.get(k, 0):.4f}" for k in all_predictions.keys()]
            })
            
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                methods = [m.split(' — ')[1] for m in all_predictions.keys()]
                pred_values = list(all_predictions.values())
                colors = ['#0f766e', '#14b8a6', '#1e7e74']
                ax.bar(methods, pred_values, color=colors, edgecolor='white', linewidth=2)
                ax.set_ylabel('Predicted Yield (Kg/Ha)', fontsize=11, color='white')
                ax.set_title('Predicted Rice Yield Comparison', fontsize=13, fontweight='bold', color='white')
                ax.set_facecolor('#1e293b')
                fig.patch.set_facecolor('#0f172a')
                plt.xticks(rotation=15, color='white', fontsize=9)
                plt.yticks(color='white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('white')
                ax.spines['bottom'].set_color('white')
                st.pyplot(fig, use_container_width=True)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                r2_vals = [all_r2_scores.get(k, 0) for k in all_predictions.keys()]
                ax.bar(methods, r2_vals, color=colors, edgecolor='white', linewidth=2)
                ax.set_ylabel('R² Score', fontsize=11, color='white')
                ax.set_title('Model Performance (R² Score)', fontsize=13, fontweight='bold', color='white')
                ax.set_ylim([0, 1])
                ax.set_facecolor('#1e293b')
                fig.patch.set_facecolor('#0f172a')
                plt.xticks(rotation=15, color='white', fontsize=9)
                plt.yticks(color='white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('white')
                ax.spines['bottom'].set_color('white')
                st.pyplot(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ **Comparison Failed:** {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# MODEL DETAILS SECTION
# ==========================================
with st.expander("📋 Model Details & Configuration", expanded=False):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Features Used in Model:**")
        feature_df = pd.DataFrame({
            "Feature": selected_features,
            "Index": range(1, len(selected_features) + 1)
        })
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("**Model Configuration:**")
        lambda_display = "None (Standard Scaling)"
        if lam == 'log':
            lambda_display = "Log Transform"
        elif isinstance(lam, (int, float)):
            lambda_display = f"{lam:.6f}"
        
        config_data = {
            "Parameter": ["Total Features", "Transformation", "Model Status", "R² Score"],
            "Value": [len(selected_features), lambda_display, "✅ Loaded", f"{r2_score_val:.4f}" if r2_score_val else "N/A"]
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
    <p>🌾 Rice Yield Prediction System | Agricultural ML Platform | v3.0</p>
    <p>Compare multiple preprocessing methods and select the best model for your data.</p>
    <p>For support or issues, please contact the system administrator.</p>
    </div>
    """,
    unsafe_allow_html=True
)