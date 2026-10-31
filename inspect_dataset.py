
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/"Data"/"student_placement_train.csv"
df=pd.read_csv(p)
print("Dataset:",p)
print("Shape:",df.shape)
print("\nColumns and dtypes:\n",df.dtypes.to_string())
print("\nMissing values:\n",df.isna().sum()[df.isna().sum()>0].to_string())
print("\nDuplicate rows:",df.duplicated().sum())
print("\nTarget counts:\n",df["placed"].value_counts(dropna=False).to_string())
