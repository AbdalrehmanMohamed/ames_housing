"""Train the house price model and save it to house_model.joblib.

Put 'AmesHousing.csv' next to this file, then run:  python train_model.py
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CSV_PATH = "AmesHousing.csv"
NUM = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "FullBath", "YearBuilt", "LotArea"]
CAT = ["Neighborhood", "KitchenQual", "CentralAir"]

df = pd.read_csv(CSV_PATH)
df.columns = [c.replace(" ", "").replace("/", "") for c in df.columns]  # same as the notebook
df = df.dropna(subset=["SalePrice"])
# Remove the known outliers (huge houses sold cheap), as in the notebook
df = df[~((df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000))]

X, y = df[NUM + CAT], np.log1p(df["SalePrice"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
])
model = Pipeline([("pre", pre), ("model", Ridge(alpha=50))])  # alpha=50 found by GridSearchCV in the notebook
model.fit(X_train, y_train)

pred = model.predict(X_test)
r2 = r2_score(y_test, pred)
mae = float(mean_absolute_error(np.expm1(y_test), np.expm1(pred)))
print(f"R2 (log) = {r2:.4f}   MAE = ${mae:,.0f}")

bundle = {
    "model": model,
    "r2": float(r2),
    "mae": mae,
    "num": NUM,
    "cat": CAT,
    "options": {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in CAT},
    "ranges": {c: (int(df[c].min()), int(df[c].max())) for c in NUM},
    "defaults": {**{c: int(df[c].median()) for c in NUM},
                 "Neighborhood": df["Neighborhood"].mode().iloc[0],
                 "KitchenQual": df["KitchenQual"].mode().iloc[0],
                 "CentralAir": df["CentralAir"].mode().iloc[0]},
}
joblib.dump(bundle, "house_model.joblib", compress=3)
print("Saved house_model.joblib")
