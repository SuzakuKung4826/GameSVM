"""
================================================================
โมเดล SVM สำหรับทำนาย "แนวเกม (Genre)" ที่เหมาะกับผู้เล่น
จากดาต้าเซ็ต Video Game Sales with Ratings (Kaggle)
================================================================
กระบวนการ Machine Learning แบบครบวงจร:
  Step 1: โหลดและทำความสะอาดข้อมูล (Data Cleaning)
  Step 2: เลือก Features / Target (Feature Selection)
  Step 3: แบ่งข้อมูล Train/Test (Data Splitting)
  Step 4: สร้าง Preprocessing Pipeline (Encoding + Scaling)
  Step 5: เทรนโมเดล SVM (Model Training)
  Step 6: ประเมินผลโมเดล (Model Evaluation)
  Step 7: บันทึกโมเดล (Model Persistence) เพื่อนำไปใช้ใน Streamlit
================================================================
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

RANDOM_STATE = 42

# ----------------------------------------------------------------
# Step 1: โหลดและทำความสะอาดข้อมูล (Data Cleaning)
# ----------------------------------------------------------------
print("=" * 60)
print("STEP 1: โหลดและทำความสะอาดข้อมูล")
print("=" * 60)

df = pd.read_csv("/mnt/user-data/uploads/Video_Games_Sales_as_at_22_Dec_2016.csv")

# คอลัมน์ User_Score เก็บเป็น string (มีค่า 'tbd' ปนอยู่) ต้องแปลงเป็นตัวเลขก่อน
df["User_Score"] = pd.to_numeric(df["User_Score"], errors="coerce")

# คอลัมน์ที่จะใช้เป็น Feature และ Target
FEATURE_COLS = ["Platform", "Rating", "Critic_Score", "User_Score", "Global_Sales"]
TARGET_COL = "Genre"

df_model = df[FEATURE_COLS + [TARGET_COL]].copy()

# ตัดแถวที่มีค่าว่าง (missing) ในคอลัมน์สำคัญออก
before = len(df_model)
df_model = df_model.dropna(subset=FEATURE_COLS + [TARGET_COL])
after = len(df_model)
print(f"จำนวนแถวก่อนตัดข้อมูลว่าง: {before}")
print(f"จำนวนแถวหลังตัดข้อมูลว่าง: {after}")

# ตัดค่า Rating ที่พบน้อยมาก (EC, K-A, RP, AO) ออก เพื่อไม่ให้รบกวนโมเดล
df_model = df_model[df_model["Rating"].isin(["E", "E10+", "T", "M"])]
print(f"จำนวนแถวหลังกรอง Rating ที่พบน้อย: {len(df_model)}")

# ----------------------------------------------------------------
# Step 2: เลือก Features / Target
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: กำหนด Features (X) และ Target (y)")
print("=" * 60)

X = df_model[FEATURE_COLS]
y = df_model[TARGET_COL]

CATEGORICAL_FEATURES = ["Platform", "Rating"]
NUMERIC_FEATURES = ["Critic_Score", "User_Score", "Global_Sales"]

print("Categorical features:", CATEGORICAL_FEATURES)
print("Numeric features:", NUMERIC_FEATURES)
print("Target classes:", sorted(y.unique()))

# ----------------------------------------------------------------
# Step 3: แบ่งข้อมูล Train / Test
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: แบ่งข้อมูล Train/Test (80/20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train set: {X_train.shape[0]} แถว")
print(f"Test set : {X_test.shape[0]} แถว")

# ----------------------------------------------------------------
# Step 4: สร้าง Preprocessing Pipeline
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: สร้าง Preprocessing Pipeline")
print("=" * 60)
print("- Numeric features -> StandardScaler (SVM ไวต่อ scale ของข้อมูลมาก)")
print("- Categorical features -> OneHotEncoder")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
)

# ----------------------------------------------------------------
# Step 5: เทรนโมเดล SVM (พร้อม Grid Search หา hyperparameter ที่ดีที่สุด)
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: เทรนโมเดล SVM")
print("=" * 60)

svm_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", SVC(probability=True, random_state=RANDOM_STATE, class_weight="balanced")),
])

# ค้นหา hyperparameter ที่ดีที่สุดด้วย GridSearchCV
param_grid = {
    "classifier__C": [1, 10],
    "classifier__kernel": ["rbf"],
    "classifier__gamma": ["scale"],
}

print("กำลังค้นหา hyperparameter ที่ดีที่สุดด้วย GridSearchCV (cv=3)...")
grid_search = GridSearchCV(svm_pipeline, param_grid, cv=3, n_jobs=-1, scoring="accuracy", verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# ----------------------------------------------------------------
# Step 6: ประเมินผลโมเดล
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: ประเมินผลโมเดลบน Test set")
print("=" * 60)

y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy บน Test set: {acc:.4f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------------------------------------------
# Step 7: บันทึกโมเดล (Model Persistence)
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: บันทึกโมเดลเป็นไฟล์ .pkl")
print("=" * 60)

joblib.dump(best_model, "/home/claude/game_svm/genre_svm_model.pkl")
print("บันทึกโมเดลเรียบร้อย: genre_svm_model.pkl")

# บันทึกข้อมูลตัวอย่างเกมในแต่ละ Genre ไว้ใช้แสดงผลใน Streamlit ด้วย
sample_games = (
    df.dropna(subset=["Genre", "Name"])
    .sort_values("Global_Sales", ascending=False)
    .groupby("Genre")
    .head(5)[["Name", "Genre", "Platform", "Global_Sales"]]
)
sample_games.to_csv("/home/claude/game_svm/sample_games_by_genre.csv", index=False)
print("บันทึกตัวอย่างเกมแต่ละแนวเรียบร้อย: sample_games_by_genre.csv")

print("\nเสร็จสิ้นกระบวนการ Machine Learning ทั้งหมด!")
