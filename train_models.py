"""
train_models.py
================
Machine Learning Assignment - 2
M.Tech (AIML) - BITS Pilani

This script:
1. Downloads / loads the Online Shoppers Purchasing Intention Dataset (UCI Repository).
2. Performs data preprocessing (StandardScaler for numerical features, OneHotEncoder for categorical features).
3. Splits data into 80% train and 20% test sets (stratified).
4. Exports `test_data.csv` containing input features and the true target column 'Revenue'.
5. Trains 5 classification models:
   - Logistic Regression
   - Decision Tree Classifier
   - K-Nearest Neighbor Classifier (KNN)
   - Naive Bayes Classifier (GaussianNB)
   - Support Vector Machine (SVM)
6. Computes 6 mandatory evaluation metrics:
   - Accuracy
   - AUC Score
   - Precision
   - Recall
   - F1 Score
   - Matthews Correlation Coefficient (MCC Score)
7. Saves all trained models and preprocessing artifacts into `model/`.
"""

import os
import json
import io
import urllib.request
import zipfile
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

def download_and_load_data():
    """Download the dataset from UCI ML Repository if not present locally."""
    data_dir = 'data'
    csv_path = os.path.join(data_dir, 'online_shoppers_intention.csv')
    
    if not os.path.exists(csv_path):
        os.makedirs(data_dir, exist_ok=True)
        print("Downloading Online Shoppers Purchasing Intention Dataset from UCI...")
        url = 'https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
            z.extractall(data_dir)
        print("Dataset extracted successfully to data/")
    else:
        print("Dataset found locally at data/online_shoppers_intention.csv")
        
    df = pd.read_csv(csv_path)
    return df

def main():
    print("=" * 70)
    print(" BITS Pilani - M.Tech (AIML) - ML Assignment 2: Model Training Pipeline ")
    print("=" * 70)
    
    # 1. Load dataset
    df = download_and_load_data()
    print(f"Total instances: {df.shape[0]}, Total attributes: {df.shape[1]}")
    
    X = df.drop(columns=['Revenue'])
    y = df['Revenue'].astype(int)
    
    # 2. Stratified 80-20 Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training instances: {X_train.shape[0]}, Test instances: {X_test.shape[0]}")
    
    # 3. Export test_data.csv (contains input features and target 'Revenue')
    test_df = X_test.copy()
    test_df['Revenue'] = y_test.values
    test_df.to_csv('test_data.csv', index=False)
    print("-> Exported 'test_data.csv' successfully.")
    
    # 4. Feature preprocessing pipeline
    cat_cols = ['Month', 'VisitorType', 'Weekend', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']
    num_cols = [c for c in X.columns if c not in cat_cols]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    # Fit preprocessor on training data only
    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Save preprocessor and metadata
    os.makedirs('model', exist_ok=True)
    joblib.dump(preprocessor, 'model/preprocessor.joblib')
    metadata = {
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'feature_names': list(X.columns),
        'target_col': 'Revenue',
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    joblib.dump(metadata, 'model/model_metadata.joblib')
    print("-> Saved 'model/preprocessor.joblib' and 'model/model_metadata.joblib'.")
    
    # 5. Define all 5 classification models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=9, weights='distance'),
        'Naive Bayes': GaussianNB(var_smoothing=0.08),
        'Support Vector Machine': SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    }
    
    metrics_summary = {}
    
    # 6. Train, evaluate, and save each model
    for name, model in models.items():
        print(f"\nTraining [{name}]...")
        model.fit(X_train_proc, y_train)
        
        # Save model
        fname = name.lower().replace(' ', '_') + '.joblib'
        save_path = os.path.join('model', fname)
        joblib.dump(model, save_path)
        print(f"-> Saved model to {save_path}")
        
        # Predictions & Probability
        y_pred = model.predict(X_test_proc)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test_proc)[:, 1]
        else:
            y_prob = model.decision_function(X_test_proc)
            
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        metrics_summary[name] = {
            'Accuracy': float(acc),
            'AUC': float(auc),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1': float(f1),
            'MCC': float(mcc),
            'File': fname
        }
    
    # 7. Save metrics JSON
    with open('model/metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=4)
        
    print("\n" + "=" * 70)
    print(" MODEL EVALUATION SUMMARY COMPARISON TABLE ")
    print("=" * 70)
    summary_df = pd.DataFrame(metrics_summary).T[['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]
    print(summary_df.round(4).to_string())
    print("=" * 70)
    print("All models successfully trained and artifacts saved to 'model/'.")

if __name__ == '__main__':
    main()
