# 🎓 Student Placement Prediction Using Machine Learning
### Predictive Analytics Using Machine Learning
**INT234 — Academic Project**

`Python` `Machine Learning` `Scikit-learn` `Pandas` `Jupyter Notebook`

---

## 📌 Project Overview

Student placement is an important aspect of academic and career development. Various factors, including academic performance, technical skills, internships, and student preferences, may be associated with placement outcomes.

This project applies predictive analytics and supervised machine learning classification algorithms to analyze student placement data and predict whether a student is placed or not placed.

The project involves dataset inspection, data cleaning, exploratory data analysis (EDA), feature preprocessing, model development, performance evaluation, and comparison of multiple machine learning algorithms.

## 🎯 Project Objectives

- Understand the dataset and examine student-related features.
- Analyze the distribution of placement outcomes.
- Identify missing values and perform data cleaning.
- Explore relationships between student features and placement outcomes.
- Apply appropriate data preprocessing techniques.
- Train and compare multiple machine learning classification models.
- Evaluate model performance using classification metrics.
- Analyze prediction errors using confusion matrices and ROC curves.
- Identify model limitations and possible improvements.

## 📂 Dataset Description

**Dataset:** `student_placement_train.csv`

The dataset contains 8,000 records and 28 columns, including student-related numerical and categorical features.

| Dataset Information | Details |
|---|---|
| Total Records | 8,000 |
| Total Columns | 28 |
| Predictor Features | 26 |
| Identifier | `student_id` |
| Target Variable | `placed` |
| Missing Cells | 2,178 |
| Duplicate Rows | 0 |

**Target variable: `placed`**

| Value | Placement Status |
|---|---|
| 0 | Not Placed |
| 1 | Placed |

**Class Distribution**

| Placement Status | Students | Percentage |
|---|---:|---:|
| Not Placed | 6,370 | 79.62% |
| Placed | 1,630 | 20.38% |

The dataset has an imbalanced target distribution. It is used for academic analysis and should not be interpreted as official LPU placement records.

## 🛠️ Technologies & Libraries

| Technology | Application |
|---|---|
| Python | Programming and analysis |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning and model evaluation |
| Joblib | Saving the trained model |
| Jupyter Notebook | Project implementation |
| Visual Studio Code | Development environment |

## 🤖 Machine Learning Models

The project compares the following classification algorithms:

**1. Dummy Classifier**

Used as a baseline model to establish a reference performance by predicting the majority class.

**2. Logistic Regression**

A classification algorithm that estimates the probability of placement using a logistic function.

**3. Decision Tree Classifier**

Uses feature-based decision rules to classify students into placement categories.

**4. Random Forest Classifier**

Combines multiple decision trees to generate classification predictions.

## ⚙️ Data Preprocessing

The following preprocessing techniques are implemented:

- Median imputation for numerical features.
- Most-frequent imputation for categorical features.
- StandardScaler for numerical feature scaling.
- OneHotEncoder for categorical feature encoding.
- ColumnTransformer for handling different feature types.
- Scikit-learn Pipeline for preprocessing and model training.
- Removal of `student_id` from model predictors.
- Stratified 80:20 train-test split.
- Five-fold Stratified Cross-Validation.

The preprocessing steps are integrated into machine learning pipelines to help prevent data leakage during model evaluation.

## 📊 Evaluation Metrics

The models are evaluated using multiple performance metrics.

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correct predictions |
| Precision | Measures the correctness of positive predictions |
| Recall | Measures how effectively placed students are identified |
| F1-score | Combines precision and recall |
| ROC-AUC | Measures the ability to distinguish between classes |
| Log Loss | Evaluates the quality of predicted probabilities |

Additional evaluation includes confusion matrices and ROC curves to examine model performance and prediction errors.

## 📈 Exploratory Data Analysis & Visualizations

Exploratory Data Analysis helps identify patterns, distributions, missing values, and relationships between student features and placement outcomes.

The project includes:

1. Placement Distribution
2. Missing Values Count
3. CGPA Distribution by Placement
4. Placement Rate by Preferred Domain
5. Placement Rate by Internships
6. Correlation Heatmap

## 📉 Model Evaluation & Results

The project compares four classification models using cross-validation and held-out test data.

**Held-out Test Results**

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy Classifier | 0.7962 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression | 0.5219 | 0.2014 | 0.4540 | 0.2790 | 0.4898 |
| Decision Tree | 0.4269 | 0.2048 | 0.6288 | 0.3090 | 0.5077 |
| Random Forest | 0.7894 | 0.3226 | 0.0307 | 0.0560 | 0.5250 |

**Key Findings**

- The dataset has a significant class imbalance.
- The Dummy Classifier achieves 79.62% accuracy by predicting the majority class but fails to identify placed students.
- Decision Tree achieves the highest recall (62.88%) and F1-score (0.3090) on the held-out test set.
- Random Forest achieves the highest ROC-AUC (0.5250), but its recall is low.
- Overall, the models demonstrate limited predictive performance.

The results highlight the importance of evaluating multiple metrics instead of relying on accuracy alone.

## 📁 Repository Structure

```text
student-placement-prediction/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── Student_Placement_Prediction.ipynb
├── inspect_dataset.py
├── eda.py
├── train_model.py
│
├── student_placement_train.csv
├── placement_model.joblib
│
├── eda_summary.txt
├── model_comparison.csv
├── model_cv_comparison.csv
├── model_evaluation.txt
├── project_methodology.txt
│
├── cgpa_distribution_by_placement.png
├── placement_distribution.png
├── missing_values_count.png
├── placement_rate_by_internships.png
├── placement_rate_by_preferred_domain.png
├── correlation_heatmap.png
├── roc_curves.png
│
├── confusion_matrix_dummyclassifier.png
├── confusion_matrix_logistic_regression.png
├── confusion_matrix_decision_tree.png
└── confusion_matrix_random_forest.png
```

*Note: This structure reflects the project files and filenames. It assumes all listed files are uploaded to the repository root.*

## ⚙️ Installation & Execution

### Prerequisites

- Python 3
- Visual Studio Code or another Python development environment
- Git (optional)

### Step 1: Clone the repository

```bash
git clone https://github.com/SAGGURTHIAVINASH/student-placement-prediction.git
cd student-placement-prediction
```

### Step 2: Create a virtual environment

```bash
py -m venv .venv
```

### Step 3: Activate the environment

```bash
.venv\Scripts\activate
```

### Step 4: Install required libraries

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Run the project scripts

```bash
python inspect_dataset.py
python eda.py
python train_model.py
```

### Step 6: Open the Jupyter Notebook

```bash
jupyter notebook
```

Open `Student_Placement_Prediction.ipynb` and run the notebook cells from top to bottom.

*Note: The execution commands assume the uploaded scripts and dataset use the filenames and paths shown above.*

## 🏆 Results & Conclusion

This project demonstrates the application of predictive analytics and supervised machine learning to student placement classification.

By performing data preprocessing, exploratory data analysis, model training, cross-validation, and performance evaluation, the project provides a structured comparison of multiple classification algorithms.

The Decision Tree achieved the highest recall and F1-score on the held-out test set among the evaluated models. However, the overall results indicate limited predictive performance, suggesting opportunities for further improvement through better features, class imbalance handling, and hyperparameter tuning.

## 🚀 Future Improvements

- Apply suitable class imbalance handling techniques.
- Perform hyperparameter optimization.
- Explore additional relevant features.
- Improve recall and F1-score.
- Evaluate models on additional datasets.
- Develop a user-friendly prediction interface.
- Investigate model interpretability techniques.

## 🎓 Academic Information

| Detail | Information |
|---|---|
| Project Title | Student Placement Prediction Using Machine Learning |
| Subject Code | INT234 |
| Subject | Predictive Analytics |
| Student Name | Saggurthi Avinash Matthew |
| Registration Number | 12413256 |
| University | Lovely Professional University |
| Academic Year | 2026 |

## 👨‍💻 Author

**Saggurthi Avinash Matthew**

B.Tech — Computer Science and Engineering  
Lovely Professional University

GitHub: [SAGGURTHIAVINASH](https://github.com/SAGGURTHIAVINASH)

*INT234 — Predictive Analytics Academic Project*

*Exploring data-driven insights through machine learning and predictive analytics.*
