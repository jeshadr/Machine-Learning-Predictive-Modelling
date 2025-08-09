# Breast Cancer Prediction using Machine Learning

This project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset** to predict whether a tumor is **benign** or **malignant** based on measurements of cell nuclei from digitized fine needle aspirate (FNA) images.

---

## 1) Goal of the Project
The goal of this project was to:
- Identify which features are most predictive of malignancy.
- Train machine learning models to predict diagnosis based on cell measurements.
- Evaluate models using multiple performance metrics.
- Visualize patterns and relationships in the dataset.

---

## 2) Dataset
The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/wasiqaliyasir/breast-cancer-dataset) and contains **569 samples** and **32 columns**:
- **`id`**: Identifier for each patient (removed before training)
- **`diagnosis`**: Target variable (`M` = malignant, `B` = benign)
- **30 numeric features** describing cell nuclei characteristics (mean, standard error, and worst values)
- **`Unnamed: 32`**: Empty column (removed during cleaning)

**Class distribution:**
- Benign: 357
- Malignant: 212

---

## 3) Data Processing
The preprocessing steps included:
- Dropping the `id` and `Unnamed: 32` columns.
- Encoding the target column: `M` → 1, `B` → 0.
- Ensuring all feature columns are numeric.
- Splitting data into training (80%) and testing (20%) sets.
- Using median imputation for any missing numeric values.
- Scaling numeric features using `StandardScaler`.

---

## 4) Exploratory Data Analysis (EDA)
I conducted EDA to better understand the dataset:

- **Correlation Heatmap**: Identified the top features correlated with malignancy (`concave points_worst`, `perimeter_worst`, `radius_worst`).
- **Barplot of Top Correlations**: Visualized the strength of relationship between each feature and diagnosis.
- **Boxplots & Violin Plots**: Compared feature distributions between benign and malignant cases.
- **KDE Plots**: Displayed separation in feature distributions for each class.
- **Pairplots**: Highlighted how multiple features interact to separate the classes.
- **Countplot**: Showed class balance in the dataset.

**Example visualizations:**
![Scatterplot Diagnosis](https://gyazo.com/a32d0d9ff4c3b359d1fc0516877b3d8e.png)
![Heatmap](https://i.gyazo.com/f9e63ffef31c9b3e1223792506a21632.png)

---

## 5) Model Training
I trained a baseline **Logistic Regression** classifier within a `Pipeline`:
1. **Imputation** (`SimpleImputer` with median strategy)
2. **Scaling** (`StandardScaler`)
3. **Model fitting** (`LogisticRegression` with `max_iter=5000`)

Other models (Random Forest, SVM) can be explored for comparison.

---

## 6) Model Evaluation
**Baseline Logistic Regression results:**
- Accuracy: **96.5%**
- Precision (Malignant): **0.97**
- Recall (Malignant): **0.93**
- F1-score (Malignant): **0.95**

---

## 7) Key Findings
- Features like `concave points_worst`, `perimeter_worst`, and `radius_worst` are the strongest predictors of malignancy.
- Logistic Regression achieved high accuracy with minimal tuning.
- Recall for malignant cases is strong, but false negatives should be minimized further.