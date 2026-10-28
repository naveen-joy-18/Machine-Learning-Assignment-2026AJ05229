import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="ShopperPulse AI | ML Classification Suite",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    .main-header h1 {
        color: white;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: #e0f2fe;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.09);
    }
    .metric-title {
        font-size: 0.82rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.3rem;
    }
    .winner-badge {
        background: #fef08a;
        color: #854d0e;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
        margin-left: 0.5rem;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        margin: 1.2rem 0 0.8rem 0;
        border-left: 4px solid #3b82f6;
        padding-left: 0.6rem;
    }
    .info-box {
        background: #f8fafc;
        border-left: 4px solid #0284c7;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        margin: 1rem 0;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Pretrained Artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'model')
    
    preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.joblib'))
    metadata = joblib.load(os.path.join(model_dir, 'model_metadata.joblib'))
    
    model_files = {
        'Logistic Regression': 'logistic_regression.joblib',
        'Decision Tree': 'decision_tree.joblib',
        'KNN': 'knn.joblib',
        'Naive Bayes': 'naive_bayes.joblib',
        'Support Vector Machine': 'support_vector_machine.joblib'
    }
    
    loaded_models = {}
    for name, fname in model_files.items():
        loaded_models[name] = joblib.load(os.path.join(model_dir, fname))
        
    metrics_path = os.path.join(model_dir, 'metrics_summary.json')
    with open(metrics_path, 'r') as f:
        metrics_summary = json.load(f)
        
    return preprocessor, metadata, loaded_models, metrics_summary

try:
    preprocessor, metadata, loaded_models, saved_metrics = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure the 'model/' directory exists and contains all required artifacts.")
    st.stop()

# ---------------------------------------------------------
# App Header
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h1>🛍️ Online Shoppers Purchasing Intention Classifier</h1>
            <p>M.Tech (AIML) Machine Learning Assignment 2 • BITS Pilani WILP</p>
        </div>
        <div style="text-align:right; font-size:0.85rem; color:#dbeafe;">
            <strong>Features:</strong> 17 | <strong>Target:</strong> Revenue (0/1)<br>
            <strong>UCI ML Benchmark</strong>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar: Controls & Dataset Upload
# ---------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/d/d3/BITS_Pilani-Logo.svg", width=140)
st.sidebar.markdown("### ⚙️ Evaluation Controls")

data_source = st.sidebar.radio(
    "Select Test Dataset Source:",
    ("Use Bundled Test Dataset (test_data.csv)", "Upload Custom Test Data (CSV)"),
    help="Streamlit free tier limits memory; only upload the test partition (with true 'Revenue' target column)."
)

test_df = None
if data_source == "Use Bundled Test Dataset (test_data.csv)":
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data.csv')
    if os.path.exists(default_path):
        test_df = pd.read_csv(default_path)
        st.sidebar.success(f"Loaded bundled test set: {test_df.shape[0]} rows, {test_df.shape[1]} columns.")
    else:
        st.sidebar.error("test_data.csv not found locally.")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Test CSV File",
        type=["csv"],
        help="Upload CSV containing all 17 feature attributes and the 'Revenue' column."
    )
    if uploaded_file is not None:
        try:
            test_df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Uploaded file loaded: {test_df.shape[0]} rows, {test_df.shape[1]} columns.")
        except Exception as e:
            st.sidebar.error(f"Error parsing CSV: {e}")

model_options = [
    "Decision Tree (Winner 🏆)",
    "Logistic Regression",
    "KNN",
    "Naive Bayes",
    "Support Vector Machine",
    "Compare All 5 Models (Multi-Model Benchmark)"
]

selected_model_option = st.sidebar.selectbox(
    "Select ML Model:",
    model_options,
    index=0,
    help="Select an individual model to view detailed diagnostics or compare all 5 side-by-side."
)

clean_model_name = selected_model_option.split(" (")[0]

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model Hyperparameters:**
- **Decision Tree**: `max_depth=5, min_samples_leaf=20`
- **SVM**: `kernel='rbf', C=1.0, probability=True`
- **Logistic Regression**: `C=1.0, max_iter=1000`
- **KNN**: `n_neighbors=9, weights='distance'`
- **Naive Bayes**: `GaussianNB(var_smoothing=0.08)`
""")

# ---------------------------------------------------------
# Validate Test Data
# ---------------------------------------------------------
if test_df is None:
    st.warning("Please select or upload a valid test dataset from the sidebar to proceed.")
    st.stop()

target_col = metadata.get('target_col', 'Revenue')
if target_col not in test_df.columns:
    st.error(f"The test dataset must contain the target column `{target_col}` for evaluation.")
    st.stop()

# Verify expected features
missing_features = [f for f in metadata['feature_names'] if f not in test_df.columns]
if missing_features:
    st.error(f"Uploaded dataset is missing required features: {missing_features}")
    st.stop()

X_test_raw = test_df[metadata['feature_names']].copy()
y_test = test_df[target_col].astype(int)

# Transform features using trained preprocessor
try:
    X_test_proc = preprocessor.transform(X_test_raw)
except Exception as e:
    st.error(f"Feature transformation error: {e}")
    st.stop()

# ---------------------------------------------------------
# Evaluation Function
# ---------------------------------------------------------
def evaluate_model(model_name, model_obj, X_proc, y_true):
    y_pred = model_obj.predict(X_proc)
    if hasattr(model_obj, "predict_proba"):
        y_prob = model_obj.predict_proba(X_proc)[:, 1]
    else:
        y_prob = model_obj.decision_function(X_proc)
        
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'Accuracy': acc,
        'AUC': auc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'MCC': mcc,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'cm': cm
    }

# Compute metrics for all models on current test data
all_eval_results = {}
for m_name, m_obj in loaded_models.items():
    all_eval_results[m_name] = evaluate_model(m_name, m_obj, X_test_proc, y_test)

# ---------------------------------------------------------
# Tabbed Main Application
# ---------------------------------------------------------
tabs = st.tabs([
    "📊 Model Performance & Diagnostics",
    "🏆 Multi-Model Comparison Table",
    "🔍 Test Predictions Explorer",
    "🧪 Live Visitor Prediction Sandbox"
])

# =========================================================
# TAB 1: Model Performance & Diagnostics
# =========================================================
with tabs[0]:
    if clean_model_name == "Compare All 5 Models":
        display_name = "Decision Tree"
        st.info("Currently displaying detailed diagnostics for **Decision Tree** (Winner). Switch between models in the sidebar or check Tab 2 for full comparison.")
    else:
        display_name = clean_model_name

    current_eval = all_eval_results[display_name]
    
    st.markdown(f"<div class='section-title'>Evaluation Metrics for {display_name}</div>", unsafe_allow_html=True)
    
    # Six Metric Cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Accuracy</div>
            <div class="metric-value">{current_eval['Accuracy']*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">AUC Score</div>
            <div class="metric-value">{current_eval['AUC']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Precision</div>
            <div class="metric-value">{current_eval['Precision']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Recall</div>
            <div class="metric-value">{current_eval['Recall']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">F1 Score</div>
            <div class="metric-value">{current_eval['F1']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MCC Score</div>
            <div class="metric-value">{current_eval['MCC']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualizations Row: Confusion Matrix, Classification Report, ROC Curve
    v_col1, v_col2 = st.columns([1, 1])

    with v_col1:
        st.markdown("#### 🟦 Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        cm = current_eval['cm']
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            annot_kws={"size": 14, "weight": "bold"},
            xticklabels=['No Purchase (0)', 'Purchase (1)'],
            yticklabels=['No Purchase (0)', 'Purchase (1)'],
            ax=ax_cm
        )
        ax_cm.set_xlabel('Predicted Label', fontweight='bold', fontsize=10)
        ax_cm.set_ylabel('True Label', fontweight='bold', fontsize=10)
        ax_cm.set_title(f'Confusion Matrix: {display_name}', fontsize=12, fontweight='bold', pad=10)
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with v_col2:
        st.markdown("#### 📈 ROC Curve & AUC")
        fpr, tpr, _ = roc_curve(y_test, current_eval['y_prob'])
        fig_roc, ax_roc = plt.subplots(figsize=(5.2, 4))
        ax_roc.plot(fpr, tpr, color='#2563eb', lw=2.5, label=f'{display_name} (AUC = {current_eval["AUC"]:.4f})')
        ax_roc.plot([0, 1], [0, 1], color='#94a3b8', lw=1.5, linestyle='--', label='Random Classifier (0.50)')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate', fontweight='bold', fontsize=10)
        ax_roc.set_ylabel('True Positive Rate', fontweight='bold', fontsize=10)
        ax_roc.set_title(f'ROC Curve (AUC: {current_eval["AUC"]:.4f})', fontsize=12, fontweight='bold', pad=10)
        ax_roc.legend(loc="lower right", frameon=True)
        ax_roc.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    # Detailed Classification Report Table
    st.markdown("#### 📋 Detailed Classification Report")
    report_dict = classification_report(y_test, current_eval['y_pred'], output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).T
    
    # Rename index for clarity
    index_map = {'0': 'Class 0 (No Purchase)', '1': 'Class 1 (Purchase)', 'accuracy': 'Overall Accuracy', 'macro avg': 'Macro Average', 'weighted avg': 'Weighted Average'}
    report_df.index = [index_map.get(i, i) for i in report_df.index]
    
    # Format display
    formatted_report = report_df.style.format({
        'precision': '{:.4f}',
        'recall': '{:.4f}',
        'f1-score': '{:.4f}',
        'support': '{:.0f}'
    }).background_gradient(subset=['f1-score'], cmap='Blues')
    
    st.dataframe(formatted_report, use_container_width=True)

# =========================================================
# TAB 2: Multi-Model Comparison Table
# =========================================================
with tabs[1]:
    st.markdown("<div class='section-title'>Model Comparison Table & Performance Analysis</div>", unsafe_allow_html=True)
    st.markdown("All 5 models evaluated on the same test dataset (`test_data.csv`) using identical preprocessing and stratified splits.")
    
    # Build comparison DataFrame
    comp_data = []
    for m_name, res in all_eval_results.items():
        comp_data.append({
            'ML Model Name': m_name,
            'Accuracy': res['Accuracy'],
            'AUC': res['AUC'],
            'Precision': res['Precision'],
            'Recall': res['Recall'],
            'F1': res['F1'],
            'MCC': res['MCC']
        })
    comp_df = pd.DataFrame(comp_data)
    
    # Display styled comparison table
    st.markdown("##### 📌 Comparison Table (Mandatory Section 3 - Step 5d)")
    styled_comp = comp_df.style.format({
        'Accuracy': '{:.4f}',
        'AUC': '{:.4f}',
        'Precision': '{:.4f}',
        'Recall': '{:.4f}',
        'F1': '{:.4f}',
        'MCC': '{:.4f}'
    }).highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'], color='#bbf7d0')
    
    st.dataframe(styled_comp, use_container_width=True)

    # Comparative Metric Bar Charts
    st.markdown("##### 📊 Visual Metric Comparison Across Models")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        fig_bar1, ax_b1 = plt.subplots(figsize=(6, 3.8))
        x = np.arange(len(comp_df))
        width = 0.35
        ax_b1.bar(x - width/2, comp_df['AUC'], width, label='AUC Score', color='#3b82f6')
        ax_b1.bar(x + width/2, comp_df['F1'], width, label='F1 Score', color='#10b981')
        ax_b1.set_xticks(x)
        ax_b1.set_xticklabels(comp_df['ML Model Name'], rotation=25, ha='right', fontsize=9)
        ax_b1.set_ylabel('Score', fontweight='bold')
        ax_b1.set_title('AUC Score vs F1 Score Comparison', fontweight='bold', fontsize=11)
        ax_b1.legend(loc='lower left')
        ax_b1.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig_bar1)
        plt.close(fig_bar1)

    with chart_col2:
        fig_bar2, ax_b2 = plt.subplots(figsize=(6, 3.8))
        ax_b2.bar(x - width/2, comp_df['Accuracy'], width, label='Accuracy', color='#8b5cf6')
        ax_b2.bar(x + width/2, comp_df['MCC'], width, label='MCC Score', color='#f59e0b')
        ax_b2.set_xticks(x)
        ax_b2.set_xticklabels(comp_df['ML Model Name'], rotation=25, ha='right', fontsize=9)
        ax_b2.set_ylabel('Score', fontweight='bold')
        ax_b2.set_title('Accuracy vs Matthews Corr Coef (MCC)', fontweight='bold', fontsize=11)
        ax_b2.legend(loc='lower left')
        ax_b2.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig_bar2)
        plt.close(fig_bar2)

    # Combined ROC Curves
    st.markdown("##### ⚡ Combined Multi-Model ROC Curves")
    fig_all_roc, ax_ar = plt.subplots(figsize=(8, 4.5))
    palette = ['#2563eb', '#16a34a', '#d97706', '#dc2626', '#9333ea']
    for idx, (m_name, res) in enumerate(all_eval_results.items()):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax_ar.plot(fpr, tpr, lw=2, color=palette[idx % len(palette)], label=f"{m_name} (AUC = {res['AUC']:.4f})")
    ax_ar.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Chance (0.50)')
    ax_ar.set_xlabel('False Positive Rate', fontweight='bold')
    ax_ar.set_ylabel('True Positive Rate', fontweight='bold')
    ax_ar.set_title('Receiver Operating Characteristic Comparison (All 5 Classifiers)', fontweight='bold', fontsize=12)
    ax_ar.legend(loc='lower right', frameon=True)
    ax_ar.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig_all_roc)
    plt.close(fig_all_roc)

    # Observations Table
    st.markdown("##### 📝 Model Observations Table")
    obs_data = [
        {"ML Model Name": "Logistic Regression", "Observation about model performance": "Solid linear baseline with high accuracy (88.08%) and high precision (73.66%), but low recall (35.86%) due to inability to model non-linear session threshold interactions."},
        {"ML Model Name": "Decision Tree", "Observation about model performance": "Best overall classifier. Captures hierarchical visitor engagement rules (e.g. PageValues thresholds) achieving top AUC (0.9178), top F1 (0.6457), top MCC (0.5910), and top Accuracy (89.94%)."},
        {"ML Model Name": "KNN", "Observation about model performance": "Decent precision (69.07%) and 87.51% accuracy, but suffers from metric sparsity in high-dimensional one-hot encoded space resulting in low recall (35.08%) and lower AUC (0.8101)."},
        {"ML Model Name": "Naive Bayes", "Observation about model performance": "Demonstrates high recall (65.45%), effectively discovering converters, but suffers from low precision (37.99%) and lower accuracy (78.10%) because the feature independence assumption fails on correlated session variables."},
        {"ML Model Name": "Support Vector Machine", "Observation about model performance": "Strong runner-up with 88.73% accuracy and 0.8824 AUC. The RBF kernel captures non-linear boundary separation well, maintaining 70.47% precision and 0.5149 MCC."}
    ]
    st.table(pd.DataFrame(obs_data))

    st.markdown("""
    <div class="info-box">
        <strong>🏆 Overall Winner: Decision Tree Classifier</strong><br>
        The Decision Tree classifier is the indisputable winner across AUC (0.9178), F1-Score (0.6457), Matthews Correlation Coefficient (0.5910), and Accuracy (89.94%). Web browsing and purchase intent behavior inherently follow sequential, condition-based decision steps (e.g. <em>if PageValues > threshold AND BounceRates < limit</em>), which tree-based architectures naturally represent with high interpretability and minimal distortion.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# TAB 3: Test Predictions Explorer
# =========================================================
with tabs[2]:
    st.markdown("<div class='section-title'>Test Dataset Predictions & Error Inspector</div>", unsafe_allow_html=True)
    st.markdown("Inspect individual test session predictions, prediction probabilities, and misclassifications.")
    
    insp_model = st.selectbox("Select Model for Predictions Inspection:", list(loaded_models.keys()), index=1)
    insp_eval = all_eval_results[insp_model]
    
    preview_df = X_test_raw.copy()
    preview_df['Actual_Revenue'] = y_test.values
    preview_df['Predicted_Revenue'] = insp_eval['y_pred']
    preview_df['Purchase_Probability'] = np.round(insp_eval['y_prob'], 4)
    preview_df['Status'] = np.where(preview_df['Actual_Revenue'] == preview_df['Predicted_Revenue'], '✅ Correct', '❌ Error')
    
    filter_choice = st.radio("Filter Sessions:", ["All Predictions", "Correct Predictions Only", "Misclassified / Errors Only"], horizontal=True)
    
    if filter_choice == "Correct Predictions Only":
        filtered_df = preview_df[preview_df['Actual_Revenue'] == preview_df['Predicted_Revenue']]
    elif filter_choice == "Misclassified / Errors Only":
        filtered_df = preview_df[preview_df['Actual_Revenue'] != preview_df['Predicted_Revenue']]
    else:
        filtered_df = preview_df
        
    st.markdown(f"Displaying **{len(filtered_df)}** records:")
    st.dataframe(filtered_df.head(100), use_container_width=True)
    
    # Download full predictions CSV
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Predictions as CSV",
        data=csv_bytes,
        file_name=f"{insp_model.lower().replace(' ', '_')}_predictions.csv",
        mime="text/csv"
    )

# =========================================================
# TAB 4: Live Visitor Prediction Sandbox
# =========================================================
with tabs[3]:
    st.markdown("<div class='section-title'>Interactive Visitor Session Simulator</div>", unsafe_allow_html=True)
    st.markdown("Tweak visitor engagement attributes to simulate live shopper sessions and test model inference.")
    
    with st.form("visitor_sim_form"):
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            st.markdown("##### 📄 Page Navigation Metrics")
            adm = st.number_input("Administrative Pageviews", min_value=0, max_value=30, value=2)
            adm_dur = st.number_input("Administrative Duration (sec)", min_value=0.0, max_value=3000.0, value=45.0)
            info = st.number_input("Informational Pageviews", min_value=0, max_value=30, value=1)
            info_dur = st.number_input("Informational Duration (sec)", min_value=0.0, max_value=3000.0, value=15.0)
            prod = st.number_input("Product Related Pageviews", min_value=0, max_value=100, value=18)
            prod_dur = st.number_input("Product Duration (sec)", min_value=0.0, max_value=10000.0, value=650.0)

        with col_s2:
            st.markdown("##### 📈 Google Analytics Signals")
            bounce = st.slider("Bounce Rate", min_value=0.0, max_value=0.20, value=0.01, step=0.005)
            exit_r = st.slider("Exit Rate", min_value=0.0, max_value=0.20, value=0.03, step=0.005)
            page_val = st.slider("Page Value (Key Predictor)", min_value=0.0, max_value=200.0, value=25.0, step=1.0)
            special_day = st.slider("Special Day Closeness (0 to 1)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
            month = st.selectbox("Month of Visit", ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], index=8)

        with col_s3:
            st.markdown("##### 💻 User & Context Details")
            visitor = st.selectbox("Visitor Type", ['Returning_Visitor', 'New_Visitor', 'Other'], index=0)
            weekend = st.selectbox("Weekend Session?", [False, True], index=0)
            os_sys = st.selectbox("Operating System Code", [1, 2, 3, 4, 5, 6, 7, 8], index=1)
            browser = st.selectbox("Browser Code", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], index=1)
            region = st.selectbox("Region Code", [1, 2, 3, 4, 5, 6, 7, 8, 9], index=0)
            traffic = st.selectbox("Traffic Type Code", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], index=1)

        sim_model_choice = st.selectbox("Choose Model for Inference:", list(loaded_models.keys()), index=1)
        submit_btn = st.form_submit_button("🚀 Run Live Purchase Prediction")

    if submit_btn:
        input_dict = {
            'Administrative': adm,
            'Administrative_Duration': adm_dur,
            'Informational': info,
            'Informational_Duration': info_dur,
            'ProductRelated': prod,
            'ProductRelated_Duration': prod_dur,
            'BounceRates': bounce,
            'ExitRates': exit_r,
            'PageValues': page_val,
            'SpecialDay': special_day,
            'Month': month,
            'OperatingSystems': os_sys,
            'Browser': browser,
            'Region': region,
            'TrafficType': traffic,
            'VisitorType': visitor,
            'Weekend': weekend
        }
        single_df = pd.DataFrame([input_dict])
        single_proc = preprocessor.transform(single_df)
        
        selected_model = loaded_models[sim_model_choice]
        pred = selected_model.predict(single_proc)[0]
        prob = selected_model.predict_proba(single_proc)[0, 1] if hasattr(selected_model, 'predict_proba') else 0.5
        
        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            if pred == 1:
                st.success(f"🎉 **Prediction: Purchase Likely (Revenue = 1)**\n\nEstimated Purchase Probability: **{prob*100:.2f}%**")
            else:
                st.warning(f"⚠️ **Prediction: No Purchase (Revenue = 0)**\n\nEstimated Purchase Probability: **{prob*100:.2f}%**")
                
        with res_col2:
            st.metric(label="Model Confidence / Probability", value=f"{prob*100:.2f}%")
            st.progress(float(prob))

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem;">
    <strong>BITS Pilani - Work Integrated Learning Programmes (WILP)</strong> | M.Tech (AIML) Machine Learning Assignment 2<br>
    Built with Streamlit & Scikit-Learn • Models: Logistic Regression, Decision Tree, KNN, Naive Bayes, Support Vector Machine
</div>
""", unsafe_allow_html=True)
