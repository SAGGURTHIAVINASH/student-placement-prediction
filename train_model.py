
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, ConfusionMatrixDisplay, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "student_placement_train.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"
for folder in (REPORTS, FIGURES, MODELS): folder.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    if "placed" not in df.columns: raise ValueError("Target column 'placed' is missing.")
    if not set(df["placed"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Target 'placed' must contain only 0 and 1.")
    if df["placed"].isna().any(): raise ValueError("Target column contains missing values.")
    y = df["placed"].astype(int)
    X = df.drop(columns=["placed", "student_id"], errors="ignore")
    if X.empty: raise ValueError("No predictor columns remain.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    numeric = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical = X_train.select_dtypes(exclude=np.number).columns.tolist()
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", num_pipe, numeric),
                              ("categorical", cat_pipe, categorical)])
    models = {
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10,
            class_weight="balanced", random_state=42, n_jobs=-1)
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy":"accuracy", "precision":"precision", "recall":"recall",
               "f1":"f1", "roc_auc":"roc_auc", "neg_log_loss":"neg_log_loss"}
    cv_rows, test_rows, fitted = [], [], {}
    for name, clf in models.items():
        pipe = Pipeline([("preprocessor", prep), ("classifier", clf)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring,
                                n_jobs=-1, error_score="raise")
        cv_rows.append({
            "Model":name,
            "Accuracy":scores["test_accuracy"].mean(),
            "Precision":scores["test_precision"].mean(),
            "Recall":scores["test_recall"].mean(),
            "F1-score":scores["test_f1"].mean(),
            "ROC-AUC":scores["test_roc_auc"].mean(),
            "Log Loss":-scores["test_neg_log_loss"].mean()
        })
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:,1]
        test_rows.append({
            "Model":name, "Accuracy":accuracy_score(y_test,pred),
            "Precision":precision_score(y_test,pred,zero_division=0),
            "Recall":recall_score(y_test,pred,zero_division=0),
            "F1-score":f1_score(y_test,pred,zero_division=0),
            "ROC-AUC":roc_auc_score(y_test,proba),
            "Log Loss":log_loss(y_test,pipe.predict_proba(X_test),labels=[0,1])
        })
        cm=confusion_matrix(y_test,pred,labels=[0,1])
        fig,ax=plt.subplots(figsize=(5.6,4.6))
        ConfusionMatrixDisplay(cm,display_labels=["Not Placed","Placed"]).plot(ax=ax,cmap="Blues",values_format="d")
        ax.set_title(f"Confusion Matrix - {name}")
        fig.tight_layout()
        fig.savefig(FIGURES/f"confusion_matrix_{name.lower().replace(' ','_')}.png",dpi=180)
        plt.close(fig)
    cv_df=pd.DataFrame(cv_rows); test_df=pd.DataFrame(test_rows)
    cv_df.to_csv(REPORTS/"model_cv_comparison.csv",index=False)
    test_df.to_csv(REPORTS/"model_comparison.csv",index=False)
    # Select using CV ROC-AUC (predeclared metric); test set is not used for selection.
    selected=cv_df.sort_values(["ROC-AUC","F1-score"],ascending=False).iloc[0]["Model"]
    joblib.dump(fitted[selected], MODELS/"placement_model.joblib")
    # ROC curves for models with probability estimates
    fig,ax=plt.subplots(figsize=(7,5))
    for name,pipe in fitted.items():
        fpr,tpr,_=roc_curve(y_test,pipe.predict_proba(X_test)[:,1])
        ax.plot(fpr,tpr,label=f"{name} (AUC={roc_auc_score(y_test,pipe.predict_proba(X_test)[:,1]):.3f})")
    ax.plot([0,1],[0,1],"--",label="Chance")
    ax.set(xlabel="False Positive Rate",ylabel="True Positive Rate",title="ROC Curves - Held-out Test Set")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGURES/"roc_curves.png",dpi=180); plt.close(fig)
    summary = [
        "STUDENT PLACEMENT PREDICTION - MODEL EVALUATION",
        f"Dataset: {DATA.relative_to(ROOT)}",
        f"Rows: {len(df)} | Columns: {df.shape[1]}",
        f"Training rows: {len(X_train)} | Test rows: {len(X_test)}",
        f"Target distribution: {y.value_counts().sort_index().to_dict()}",
        f"Numerical features: {len(numeric)} | Categorical features: {len(categorical)}",
        "Preprocessing: median imputation + standard scaling for numeric; most-frequent imputation + one-hot encoding for categorical.",
        "Leakage control: preprocessing is inside each Pipeline and is fitted separately within CV folds / training split.",
        "student_id is excluded; placed is used only as target.",
        f"Selected model by highest mean 5-fold CV ROC-AUC (tie-break F1): {selected}",
        "",
        "5-FOLD STRATIFIED CV RESULTS (TRAINING PARTITION ONLY)",
        cv_df.to_string(index=False, float_format=lambda x:f"{x:.4f}"),
        "",
        "HELD-OUT TEST RESULTS",
        test_df.to_string(index=False, float_format=lambda x:f"{x:.4f}"),
        "",
        "Interpretation: Accuracy must be read alongside recall, F1, ROC-AUC and Log Loss because the target classes are imbalanced.",
        "A ROC-AUC near 0.50 indicates performance close to random ranking. Results describe this dataset only and do not establish real-world placement causation."
    ]
    (REPORTS/"model_evaluation.txt").write_text("\n".join(summary),encoding="utf-8")
    print("\n".join(summary))
if __name__=="__main__": main()
