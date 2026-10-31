# Student Placement Prediction Using Machine Learning

**Course:** INT234 – Predictive Analytics  
**Project Type:** Machine Learning Classification

## Project Overview

This project focuses on predicting student placement outcomes using machine learning classification algorithms. It follows a predictive analytics workflow involving dataset inspection, data cleaning, exploratory data analysis, preprocessing, model training, and performance evaluation.

The objective is to compare different classification algorithms and understand how effectively they identify placed and not-placed students.

## Objectives

- Analyze student placement data.
- Perform data cleaning and preprocessing.
- Explore relationships between student features and placement outcomes.
- Apply multiple machine learning classification algorithms.
- Compare models using evaluation metrics.
- Identify the limitations of placement prediction.

## Dataset Information

| Description | Details |
|---|---|
| Dataset | Student Placement Dataset |
| Records | 8,000 |
| Columns | 28 |
| Target variable | `placed` |
| Not Placed | 6,370 |
| Placed | 1,630 |
| Missing cells | 2,178 |
| Duplicate rows | 0 |

**Target variable:**
- `0` – Not Placed
- `1` – Placed

The dataset contains an imbalanced target distribution. The dataset is used for academic analysis and should not be interpreted as official LPU placement records.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Joblib

## Project Workflow

1. Dataset loading and inspection.
2. Data cleaning and missing-value analysis.
3. Exploratory Data Analysis (EDA).
4. Feature preparation and preprocessing.
5. Train-test splitting.
6. Model training.
7. Cross-validation.
8. Model evaluation and comparison.
9. Visualization of results.
10. Saving the trained model.

## Machine Learning Algorithms

The following classification algorithms are implemented:

- Dummy Classifier
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

## Data Preprocessing

- Median imputation for numerical features.
- Most-frequent imputation for categorical features.
- StandardScaler for numerical features.
- OneHotEncoder for categorical features.
- ColumnTransformer and Scikit-learn Pipeline.
- Stratified 80:20 train-test split.
- Five-fold Stratified Cross-Validation.

The `student_id` column is excluded from model training.

## Exploratory Data Analysis

The following visualizations are included:

- Placement Distribution
- Missing Values Count
- CGPA Distribution by Placement
- Placement Rate by Preferred Domain
- Placement Rate by Internships
- Correlation Heatmap

## Model Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Log Loss
- Confusion Matrix
- ROC Curve

The project includes cross-validation results and held-out test-set evaluation.

### Key Findings

The dataset has an imbalanced target distribution, with 79.62% not placed and 20.38% placed.

The Dummy Classifier achieves high accuracy by predicting the majority class, but it fails to identify placed students. The Decision Tree achieves the highest recall on the held-out test set among the evaluated models. Overall, the results indicate limited predictive performance, highlighting the need for further feature engineering, data analysis, and model improvement.

## Project Files

| File | Description |
|---|---|
| `Student_Placement_Prediction.ipynb` | Complete project notebook |
| `inspect_dataset.py` | Dataset inspection |
| `eda.py` | Exploratory data analysis |
| `train_model.py` | Model training and comparison |
| `requirements.txt` | Required Python libraries |
| `student_placement_train.csv` | Dataset |
| `placement_model.joblib` | Saved trained model |
| `model_comparison.csv` | Model comparison results |
| `model_cv_comparison.csv` | Cross-validation results |
| `model_evaluation.txt` | Evaluation report |
| `eda_summary.txt` | EDA summary |
| `project_methodology.txt` | Project methodology |

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/SAGGURTHIAVINASH/student-placement-prediction.git
cd student-placement-prediction
```

### 2. Create a virtual environment

```bash
py -m venv .venv
```

### 3. Activate the environment

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the project

```bash
python inspect_dataset.py
python eda.py
python train_model.py
```

You can also open the Jupyter Notebook:

```bash
jupyter notebook
```

Open `Student_Placement_Prediction.ipynb` and run the cells.

## Limitations and Future Improvements

- Address class imbalance using appropriate techniques.
- Explore additional relevant student features.
- Perform hyperparameter tuning.
- Improve recall and F1-score.
- Evaluate models on additional datasets.
- Develop a user-friendly prediction interface.

The current results should not be treated as guaranteed real-world placement predictions or evidence of causal relationships.

## Author

**Saggurthi Avinash Matthew**  
B.Tech – Computer Science and Engineering  
Lovely Professional University  
Course: INT234 – Predictive Analytics

GitHub: [SAGGURTHIAVINASH](https://github.com/SAGGURTHIAVINASH)

---

*Academic Machine Learning Project – 2026*
