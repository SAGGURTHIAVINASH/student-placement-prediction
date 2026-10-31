# Student Placement Prediction — INT234 Predictive Analytics

## Project objective
Apply a standard predictive-analytics workflow to the supplied Kaggle student-placement CSV and compare classification models for `placed` (0 = Not Placed, 1 = Placed).

This project follows the workflow described in the LinkedIn reference project: preprocessing and cleaning, EDA, feature preparation, multiple classification models, and evaluation using Accuracy, Precision, Recall, F1-score, Log Loss and ROC-AUC, together with visualization, correlation analysis and confusion matrices.

## Dataset
- File: `Data/student_placement_train.csv`
- Shape: 8,000 rows × 28 columns
- Predictors: 26, plus `student_id` identifier
- Target: `placed`
- Class counts: 6,370 not placed; 1,630 placed
- Missing cells: 2,178 across 11 columns
- Duplicate rows: 0

The original Kaggle CSV is preserved. Missing predictor values are imputed inside the modeling pipeline; the source CSV is not overwritten. The dataset should not be described as official LPU placement records unless provenance is independently verified.

## Workflow
1. Dataset loading and validation (`src/inspect_dataset.py`)
2. Exploratory data analysis (`src/eda.py`)
3. Preprocessing: median imputation and scaling for numerical features; most-frequent imputation and one-hot encoding for categorical features
4. Feature preparation: remove `student_id` from predictors; separate `placed` as target
5. Stratified 80:20 train/test split with random_state=42
6. Five-fold stratified cross-validation on training data
7. Compare DummyClassifier, Logistic Regression, Decision Tree and Random Forest
8. Evaluate held-out test data with Accuracy, Precision, Recall, F1-score, ROC-AUC and Log Loss
9. Save confusion matrices, ROC plot, metrics, evaluation report and fitted pipeline

## Run on Windows
Open PowerShell in the extracted project folder:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src\inspect_dataset.py
python src\eda.py
python src\train_model.py
jupyter notebook
```

Open `notebooks/Student_Placement_Prediction.ipynb` and run all cells from top to bottom.

## Main files
- `Data/`: original Kaggle dataset
- `src/inspect_dataset.py`: dataset checks
- `src/eda.py`: EDA plots and summary
- `src/train_model.py`: preprocessing, CV, model comparison and saved model
- `notebooks/Student_Placement_Prediction.ipynb`: academic end-to-end notebook
- `reports/`: metric tables, text reports and figures
- `models/placement_model.joblib`: selected fitted preprocessing + classifier pipeline

## Reproducibility and limitations
- `student_id` is excluded from predictors.
- `placed` is never included in the feature matrix.
- Imputation, scaling and encoding are inside Scikit-learn pipelines, so they are fitted on training folds only during CV and on training data for final fit.
- The test partition is not used for model selection.
- The selected model is chosen using mean 5-fold CV ROC-AUC (F1-score as tie-break).
- The target is imbalanced. Accuracy by itself can be misleading.
- A ROC-AUC near 0.50 indicates weak ranking performance. Do not claim strong placement prediction or causal relationships from these results.
- Company recommendation / real-world placement guarantees are outside this reference-aligned academic classification workflow.
