# Machine Learning Assignment - 2

**Course:** Machine Learning

**Name:** Naveen Joy
**BITS-ID:** 2026AJ05229

---

## a. Problem Statement

In modern digital commerce, understanding customer browsing behavior and accurately predicting whether an online session will culminate in a purchase transaction is critical for revenue growth, dynamic personalized marketing, and conversion rate optimization (CRO).

The goal of this project is to develop, evaluate, and deploy an end-to-end Machine Learning classification system that predicts **Online Shoppers Purchasing Intention** (`Revenue`: True/False). Utilizing a multi-model comparative framework, we implement, benchmark, and deploy five distinct classification algorithms:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbor (KNN) Classifier**
4. **Naive Bayes Classifier (Gaussian NB)**
5. **Support Vector Machine (SVM with RBF Kernel)**

---

## b. Dataset Description

- **Source Repository:** UCI Machine Learning Repository / Kaggle
- **Dataset Name:** [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **Total Instances:** 12,330 sessions (Exceeds the minimum threshold of 500 instances)
- **Total Features:** 17 input features + 1 binary target feature = 18 attributes (Exceeds the minimum threshold of 12 features)
- **Classification Task:** Binary Classification
- **Target Variable:** `Revenue` (Boolean: `True` indicates purchase completed, `False` indicates no purchase)
  - Class Distribution: ~84.5% `False` (10,422 sessions), ~15.5% `True` (1,908 sessions), reflecting real-world e-commerce class imbalance.

### Feature Dictionary

| Feature Name              | Type        | Description                                                                         |
| :------------------------ | :---------- | :---------------------------------------------------------------------------------- |
| `Administrative`          | Numeric     | Number of pages visited by the user in the administrative category                  |
| `Administrative_Duration` | Numeric     | Total time spent (seconds) on administrative pages                                  |
| `Informational`           | Numeric     | Number of pages visited by the user in the informational category                   |
| `Informational_Duration`  | Numeric     | Total time spent (seconds) on informational pages                                   |
| `ProductRelated`          | Numeric     | Number of pages visited by the user in the product category                         |
| `ProductRelated_Duration` | Numeric     | Total time spent (seconds) on product-related pages                                 |
| `BounceRates`             | Numeric     | Average bounce rate of pages visited (percentage of single-page sessions)           |
| `ExitRates`               | Numeric     | Average exit rate of pages visited (percentage of exits from page)                  |
| `PageValues`              | Numeric     | Average Google Analytics page value for pages visited prior to transaction          |
| `SpecialDay`              | Numeric     | Closeness of site visiting time to a specific holiday or event (e.g., Mother's Day) |
| `Month`                   | Categorical | Month of the visit (Feb, Mar, May, June, Jul, Aug, Sep, Oct, Nov, Dec)              |
| `OperatingSystems`        | Categorical | Operating system identifier code (1–8)                                              |
| `Browser`                 | Categorical | Browser identifier code (1–13)                                                      |
| `Region`                  | Categorical | Geographic region identifier code (1–9)                                             |
| `TrafficType`             | Categorical | Web traffic source code (1–20)                                                      |
| `VisitorType`             | Categorical | Visitor category (`Returning_Visitor`, `New_Visitor`, `Other`)                      |
| `Weekend`                 | Boolean     | Binary flag indicating whether the visit occurred on a weekend                      |
| `Revenue` _(Target)_      | Boolean     | Ground truth purchase flag (`True` = 1, `False` = 0)                                |

---

## c. GitHub Repository Link

- **GitHub Repository:** (https://github.com/naveen-joy-18/Machine-Learning-Assignment-2026AJ05229.git)
- **Live Streamlit App:** (https://machine-learning-assignment-2026aj05229git-ikh5op7spzy4wus2bfj.streamlit.app/)

### Repository Structure

```text
project-folder/
│-- app.py                  # Interactive Streamlit Web Application
│-- requirements.txt        # Python dependency manifest for local & cloud runtime
│-- README.md               # Detailed academic project documentation
│-- test_data.csv           # Stratified test split (2,466 rows) with true Revenue labels
   │-- train_models.py         # End-to-end reproducible training pipeline script
│-- model/                  # Directory containing serialized models & artifacts
│   │-- logistic_regression.joblib
│   │-- decision_tree.joblib
│   │-- knn.joblib
│   │-- naive_bayes.joblib
│   │-- support_vector_machine.joblib
│   │-- preprocessor.joblib  # Fitted ColumnTransformer (StandardScaler + OneHotEncoder)
│   │-- model_metadata.joblib # Feature names, categorical/numerical splits
│   │-- metrics_summary.json # Serialized evaluation metrics dictionary
│   │-- model_training.ipynb # Jupyter Notebook with visual training workflow
│   │-- train_models.py      # Duplicate source script inside model directory
```

---

## d. Models Used & Comparative Evaluation

All five classification models were trained on the exact same 80% stratified training partition (9,864 samples) and evaluated on the identical 20% holdout test partition (2,466 samples) using standardized preprocessing (`StandardScaler` for continuous features, `OneHotEncoder` with `handle_unknown='ignore'` for categorical features).

### Model Evaluation Comparison Table

| ML Model Name              |  Accuracy  |    AUC     | Precision |   Recall   |     F1     |    MCC     |
| :------------------------- | :--------: | :--------: | :-------: | :--------: | :--------: | :--------: |
| **Logistic Regression**    |   0.8808   |   0.8828   |  0.7366   |   0.3586   |   0.4824   |   0.4592   |
| **Decision Tree**          | **0.8994** | **0.9178** |  0.7107   |   0.5916   | **0.6457** | **0.5910** |
| **KNN**                    |   0.8751   |   0.8101   |  0.6907   |   0.3508   |   0.4653   |   0.4327   |
| **Naive Bayes**            |   0.7810   |   0.8065   |  0.3799   | **0.6545** |   0.4808   |   0.3752   |
| **Support Vector Machine** |   0.8873   |   0.8824   |  0.7047   |   0.4686   |   0.5629   |   0.5149   |

---

### Model Observations Table

| ML Model Name              | Observation about model performance                                                                                                                                                                                                                                                                                                                         |
| :------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**    | Serves as a strong linear baseline with high accuracy (88.08%) and strong precision (73.66%). However, it suffers from a low recall of 35.86%, indicating that a linear hyperplane cannot adequately capture complex non-linear combinations of engagement metrics such as high page value with low bounce rate.                                            |
| **Decision Tree**          | Best performing model overall. Achieves the highest AUC (0.9178), F1 score (0.6457), Matthews Correlation Coefficient (0.5910), and Accuracy (89.94%). Web browsing purchase intent follows hierarchical decision rules (e.g., threshold conditions on `PageValues` and `ProductRelated_Duration`), which tree structures model naturally.                  |
| **KNN**                    | Demonstrates solid precision (69.07%) and 87.51% accuracy, but suffers from distance metric degradation (curse of dimensionality) in the high-dimensional transformed feature space created by one-hot encoding, resulting in modest recall (35.08%) and an AUC of 0.8101.                                                                                  |
| **Naive Bayes**            | Achieves the highest recall among standard baselines (65.45%), effectively flagging potential purchasers. However, it exhibits poor precision (37.99%) and lower overall accuracy (78.10%) because the core assumption of class-conditional feature independence is strongly violated by correlated session parameters (such as page counts and durations). |
| **Support Vector Machine** | Strong runner-up with 88.73% accuracy and 0.8824 AUC. The RBF kernel successfully captures non-linear boundary separation in the multi-dimensional feature space, balancing precision (70.47%) and recall (46.86%) with an MCC of 0.5149.                                                                                                                   |

---

### Overall Report for the Dataset

> 🏆 **Overall High Metrics: Decision Tree Classifier**

#### Analytical Rationale:

1. **Dominant Discriminative Power (AUC = 0.9178):** The Decision Tree achieved an AUC of 0.9178, surpassing all linear and kernel-based baselines, demonstrating superior capability in ranking positive purchase sessions above non-purchase sessions across all operating thresholds.
2. **Balanced Imbalanced-Class Performance (F1 = 0.6457, MCC = 0.5910):** In class-imbalanced settings (~15.5% minority positive class), raw Accuracy can be deceptive. The Decision Tree attained both the highest F1 score (0.6457) and the highest Matthews Correlation Coefficient (0.5910), proving it delivers genuine balanced predictive power without biasing toward the majority negative class.
3. **Alignment with Behavioral Dynamics:** Consumer decision journeys in e-commerce are step-wise and threshold-driven (e.g., `PageValues > 18.5` and `BounceRates < 0.02`). The hierarchical partitioning of Decision Trees mirrors this underlying business process seamlessly, avoiding boundary distortions found in linear or metric-distance algorithms.

---

## e. Local Setup and Deployment Guide

### 1. Local Execution

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd <repo-folder>
   ```
2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
5. Open your browser at `http://localhost:8501`.

### 2. Streamlit Community Cloud Deployment

1. Push your complete code to a public GitHub repository.
2. Visit [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in with your GitHub account.
3. Click **"New App"**.
4. Select your repository, specify the branch (usually `main`), and set the main file path to `app.py`.
5. Click **"Deploy"**. The app will be live within 2 minutes!

---

## f. Academic Integrity & Anti-Plagiarism Compliance

- **Independent Implementation:** Dataset selection, feature engineering pipelines, model hyperparameter tuning, evaluation matrices, and Streamlit frontend styling were developed from first principles.
- **Reproducibility:** All artifacts are pre-trained and serialized using `joblib`. The training script `train_models.py` and interactive notebook `model/model_training.ipynb` provide complete transparency and 100% deterministic reproducibility.
