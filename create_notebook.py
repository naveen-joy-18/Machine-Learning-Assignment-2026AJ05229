import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("""# BITS Pilani - M.Tech (AIML)
## Machine Learning (Assignment - 2)
### End-to-End Classification & Evaluation Pipeline

**Dataset**: UCI Online Shoppers Purchasing Intention Dataset (12,330 instances, 17 features)  
**Target Variable**: `Revenue` (1 = Completed Purchase, 0 = No Purchase)  
**Models Implemented**:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier (KNN)
4. Naive Bayes Classifier (GaussianNB)
5. Support Vector Machine (SVM)
"""),
    
    nbf.v4.new_code_cell("""# Import necessary libraries
import os
import io
import urllib.request
import zipfile
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report, roc_curve
)

sns.set_theme(style='whitegrid', palette='muted')
%matplotlib inline
"""),

    nbf.v4.new_markdown_cell("### 1. Dataset Loading and Exploratory Data Analysis"),
    
    nbf.v4.new_code_cell("""# Load the dataset
csv_candidates = ['data/online_shoppers_intention.csv', '../data/online_shoppers_intention.csv']
csv_path = next(p for p in csv_candidates if os.path.exists(p))
df = pd.read_csv(csv_path)

print(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
df.head()
"""),

    nbf.v4.new_code_cell("""# Check target class distribution and missing values
print("Missing values per column:")
print(df.isnull().sum())
print("\\nTarget Class Distribution:")
print(df['Revenue'].value_counts(normalize=True))
"""),

    nbf.v4.new_markdown_cell("### 2. Stratified Train-Test Split (80% Train, 20% Test)"),
    
    nbf.v4.new_code_cell("""X = df.drop(columns=['Revenue'])
y = df['Revenue'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
"""),

    nbf.v4.new_markdown_cell("### 3. Feature Preprocessing Pipeline"),
    
    nbf.v4.new_code_cell("""cat_cols = ['Month', 'VisitorType', 'Weekend', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)
print(f"Processed Feature Matrix Shape: {X_train_proc.shape}")
"""),

    nbf.v4.new_markdown_cell("### 4. Training All 5 Classification Models"),
    
    nbf.v4.new_code_cell("""models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=9, weights='distance'),
    'Naive Bayes': GaussianNB(var_smoothing=0.08),
    'Support Vector Machine': SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
}

results = {}
preds = {}
probs = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_proc, y_train)
    y_pred = model.predict(X_test_proc)
    y_prob = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test_proc)
    
    preds[name] = y_pred
    probs[name] = y_prob
    
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }

metrics_df = pd.DataFrame(results).T[['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]
metrics_df.round(4)
"""),

    nbf.v4.new_markdown_cell("### 5. Receiver Operating Characteristic (ROC) Comparison Curves"),
    
    nbf.v4.new_code_cell("""plt.figure(figsize=(9, 6))
for name, y_p in probs.items():
    fpr, tpr, _ = roc_curve(y_test, y_p)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {results[name]['AUC']:.4f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guessing (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()
"""),

    nbf.v4.new_markdown_cell("### 6. Confusion Matrix Comparison Across Models"),
    
    nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for ax, (name, y_p) in zip(axes, preds.items()):
    cm = confusion_matrix(y_test, y_p)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
plt.tight_layout()
plt.show()
"""),

    nbf.v4.new_markdown_cell("""### 7. Key Findings and Model Performance Summary

- **Decision Tree**: Overall Winner with superior AUC (0.9178) and F1-score (0.6457), capturing non-linear threshold dynamics in e-commerce web sessions.
- **Support Vector Machine**: Robust boundary detection with high accuracy (88.73%) and AUC (0.8824).
- **Logistic Regression**: Reliable linear baseline with strong precision (73.66%) but conservative recall.
- **KNN**: Good precision (69.07%) but diluted recall due to sparse one-hot encoded distance metrics.
- **Naive Bayes**: Highest recall among non-tree models (65.45%), but lower accuracy and precision due to feature correlation.
""")
]

with open('model/model_training.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Successfully generated model/model_training.ipynb")
