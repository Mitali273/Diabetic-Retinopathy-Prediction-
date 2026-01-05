import pandas as pd
import os # for handling directories and file patth
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb


DATA_PATH = r"C:\Ankit_Singh\Data Science\Sample Data\pronostico_dataset.csv"
ARTIFACTS_DIR = "artifacts" # creating a folder to save model and output
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


df = pd.read_csv(DATA_PATH)
print("Initial Data Shape:", df.shape)
print("Columns:", df.columns)


df["prognosis"] = df["prognosis"].map({"no_retinopathy": 0, "retinopathy": 1})

X = df.drop(columns=["ID", "prognosis"], errors="ignore")
y = df["prognosis"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True),
    "XGBoost": xgb.XGBClassifier(eval_metric="logloss"),
    "LightGBM": lgb.LGBMClassifier()
}

metrics = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    metrics[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    joblib.dump(model, os.path.join(ARTIFACTS_DIR, f"{name.replace(' ', '_').lower()}.pkl"))


metrics_df = pd.DataFrame(metrics).T
metrics_df.to_csv(os.path.join(ARTIFACTS_DIR, "metrics.csv"))
print("\nTraining completed. Metrics saved at:", os.path.join(ARTIFACTS_DIR, "metrics.csv"))
