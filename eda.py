
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/"Data"/"student_placement_train.csv")
out=ROOT/"reports"/"figures"; out.mkdir(parents=True,exist_ok=True)
sns.set_theme(style="whitegrid")
# Target distribution
fig,ax=plt.subplots(figsize=(6,4)); sns.countplot(data=df,x="placed",ax=ax,color="steelblue")
ax.set_xticks([0,1],["Not Placed (0)","Placed (1)"]); ax.set_title("Placement Class Distribution")
fig.tight_layout(); fig.savefig(out/"placement_distribution.png",dpi=180); plt.close(fig)
# Missing values
miss=df.isna().sum().sort_values(ascending=False); miss=miss[miss>0]
fig,ax=plt.subplots(figsize=(8,max(4,len(miss)*.35))); miss.sort_values().plot(kind="barh",ax=ax)
ax.set_title("Missing Values by Feature"); ax.set_xlabel("Missing cell count")
fig.tight_layout(); fig.savefig(out/"missing_values_count.png",dpi=180); plt.close(fig)
# CGPA distribution
if "cgpa" in df:
 fig,ax=plt.subplots(figsize=(7,4)); sns.histplot(data=df,x="cgpa",hue="placed",kde=True,stat="density",common_norm=False,ax=ax)
 ax.set_title("CGPA Distribution by Placement"); fig.tight_layout(); fig.savefig(out/"cgpa_distribution_by_placement.png",dpi=180); plt.close(fig)
# Placement rate by preferred domain
if "preferred_domain" in df:
 rates=df.groupby("preferred_domain",dropna=False)["placed"].mean().sort_values(ascending=False)
 fig,ax=plt.subplots(figsize=(8,4)); rates.plot(kind="bar",ax=ax); ax.set_ylabel("Placement rate"); ax.set_title("Placement Rate by Preferred Domain")
 fig.tight_layout(); fig.savefig(out/"placement_rate_by_preferred_domain.png",dpi=180); plt.close(fig)
# Placement rate by internships
if "internships_count" in df:
 rates=df.groupby("internships_count")["placed"].mean()
 fig,ax=plt.subplots(figsize=(7,4)); rates.plot(marker="o",ax=ax); ax.set_ylabel("Placement rate"); ax.set_title("Placement Rate by Internship Count")
 fig.tight_layout(); fig.savefig(out/"placement_rate_by_internships.png",dpi=180); plt.close(fig)
# Correlation heatmap
num=df.select_dtypes(include="number").drop(columns=["student_id"],errors="ignore")
fig,ax=plt.subplots(figsize=(12,9)); sns.heatmap(num.corr(),cmap="coolwarm",center=0,ax=ax)
ax.set_title("Correlation Heatmap"); fig.tight_layout(); fig.savefig(out/"correlation_heatmap.png",dpi=180); plt.close(fig)
summary=[
 f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns",
 f"Duplicate rows: {df.duplicated().sum()}",
 f"Total missing cells: {int(df.isna().sum().sum())}",
 f"Target counts: {df['placed'].value_counts().sort_index().to_dict()}",
 f"Target proportions: {df['placed'].value_counts(normalize=True).sort_index().round(4).to_dict()}",
 "Missing values by column:", df.isna().sum()[df.isna().sum()>0].to_string(),
 "Numeric correlations with target:", num.corr()['placed'].drop('placed').sort_values(key=abs,ascending=False).head(10).to_string()
]
(ROOT/"reports"/"eda_summary.txt").write_text("\n".join(summary),encoding="utf-8")
print("\n".join(summary))
