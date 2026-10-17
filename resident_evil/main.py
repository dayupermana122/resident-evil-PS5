from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lars
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Folder program dan lokasi dataset
FOLDER = Path(__file__).resolve().parent
DATA = FOLDER / "resident_evil_ps5.csv"
OUT = FOLDER / "output"
OUT.mkdir(exist_ok=True)

# 1. DATA LOADING
df = pd.read_csv(DATA)
print("\\n=== 1. DATA LOADING ===")
print("Ukuran dataset:", df.shape)
print("\\n5 baris pertama:")
print(df.head())
print("\\nInformasi dataset:")
df.info()
print("\\nStatistik deskriptif:")
print(df.describe(include="all").T)
print("\\nMissing values:")
print(df.isnull().sum())
print("\\nJumlah duplikat:", df.duplicated().sum())

# 2. DATA CLEANING
df = df.drop_duplicates().copy()
target = "Total_Copies_Sold_Millions"
numeric_cols = [
    "Release_Year", "Metacritic_Score", "User_Score",
    "Launch_Price_USD", "Game_Length_Hours"
]
categorical_cols = ["Format_Type", "Perspective"]
features = numeric_cols + categorical_cols

# Baris tanpa target tidak bisa dipakai untuk melatih model
df = df.dropna(subset=[target]).copy()
print("\\n=== 2. DATA CLEANING ===")
print("Ukuran setelah membersihkan duplikat/target kosong:", df.shape)

# 3. EDA: simpan grafik ke folder output
print("\\n=== 3. EDA ===")
df[numeric_cols + [target]].hist(figsize=(12, 9), bins=10, edgecolor="black")
plt.suptitle("Histogram Fitur Numerik dan Total Kopi Terjual")
plt.tight_layout()
plt.savefig(OUT / "01_histogram.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 5))
sns.boxplot(data=df[numeric_cols])
plt.title("Boxplot Fitur Numerik")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(OUT / "02_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()

corr = df[numeric_cols + [target]].corr(numeric_only=True)
plt.figure(figsize=(9, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Heatmap Korelasi")
plt.tight_layout()
plt.savefig(OUT / "03_heatmap_korelasi.png", dpi=150, bbox_inches="tight")
plt.close()
print("Grafik tersimpan di folder output.")
print("\\nKorelasi fitur numerik dengan target:")
print(corr[target].sort_values(ascending=False))

# 4. DATA SPLITTING
print("\\n=== 4. DATA SPLITTING ===")
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print("Total data:", len(df))
print("Data latih:", len(X_train))
print("Data uji:", len(X_test))

# 5. PREPROCESSING + MODELLING
print("\\n=== 5. PREPROCESSING DAN MODELLING ===")
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoding", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

models = {
    "LARS": Lars(),
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42
    )
}

results = []
pipelines = {}
for name, estimator in models.items():
    pipe = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", estimator)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)
    results.append({
        "Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2
    })
    pipelines[name] = pipe
    print(f"\\n{name}")
    print(f"MAE  : {mae:.4f} juta kopi")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f} juta kopi")
    print(f"R2   : {r2:.4f}")

results_df = pd.DataFrame(results).sort_values("MAE")
results_df.to_csv(OUT / "hasil_evaluasi_model.csv", index=False)
print("\\nRingkasan evaluasi:")
print(results_df.to_string(index=False))

# 6. SIMPAN MODEL DAN CONTOH PREDIKSI
best_name = results_df.iloc[0]["Model"]
best_model = pipelines[best_name]
joblib.dump(best_model, OUT / "model_terbaik.joblib")

comparison = X_test.copy()
comparison["Aktual_juta_kopi"] = y_test
comparison["Prediksi_juta_kopi"] = best_model.predict(X_test)
comparison.to_csv(OUT / "perbandingan_prediksi.csv", index=False)

print("\\n=== 6. HASIL AKHIR ===")
print("Model dengan MAE terkecil pada percobaan ini:", best_name)
print("File model: output/model_terbaik.joblib")
print("Tabel evaluasi: output/hasil_evaluasi_model.csv")
print("Perbandingan prediksi: output/perbandingan_prediksi.csv")
print("SELESAI. Silakan buka folder output untuk melihat grafik dan hasil.")
