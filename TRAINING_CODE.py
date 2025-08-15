# exo_life_pipeline.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# -------------------------
# 1. Load Dataset
# -------------------------
df = pd.read_csv("C:\\Users\\kesha\\Downloads\\pl_data.csv")

# Keep only relevant numeric columns
features = ['pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_bmasse',
            'pl_insol', 'pl_eqt', 'st_teff', 'st_rad', 'st_lum', 'habitable']
df["habitable"] = (
    (df["pl_eqt"] >= 180) & (df["pl_eqt"] <= 310) &
    (df["pl_rade"] >= 0.5) & (df["pl_rade"] <= 2.5) &
    (df["st_teff"] > 3000) & (df["st_teff"] < 7000)
).astype(int)
df = df[features]



# Drop missing values
df.dropna(inplace=True)

# -------------------------
# 2. Split features/target
# -------------------------
X = df.drop(columns=['habitable'])
y = df['habitable']

# -------------------------
# 3. Handle imbalance with SMOTE
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline: SMOTE + Scaling + Model
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=200, random_state=42))
])

# -------------------------
# 4. Train the model
# -------------------------
pipeline.fit(X_train, y_train)

# -------------------------
# 5. Evaluate
# -------------------------
y_pred = pipeline.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------
# 6. Save the full pipeline
# -------------------------
joblib.dump(pipeline, "exo_life_pipeline.pkl")
print("✅ Model + preprocessing pipeline saved as 'exo_life_pipeline.pkl'")
