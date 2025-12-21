import os
import shutil # untuk pembersihan
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# KONFIGURASI ENV VAR (Sesuai CI/CD)
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/AllsHub/Submission_MLOps_aldomp7.mlflow"
# User & Pass diambil dari Secrets GitHub Actions

def main():
    mlflow.set_experiment("Breast Cancer Classification - CI Workflow")
    
    with mlflow.start_run():
        # --- 1. Load Data ---
        csv_filename = 'breastcancer_preprocessing.csv' 
        if not os.path.exists(csv_filename):
            print(f"Error: File {csv_filename} tidak ditemukan.")
            return
        
        df = pd.read_csv(csv_filename)
        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # --- 2. Training ---
        rf = RandomForestClassifier(random_state=42)
        params = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
        clf = GridSearchCV(rf, params, cv=3, scoring='accuracy')
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_
        
        # --- 3. Evaluasi ---
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_params(clf.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # --- 4. SAVE & UPLOAD ---
        signature = infer_signature(X_train, y_pred)
        
        # Nama folder output yang jelas
        local_model_path = "model_output"
        
        # Hapus dulu jika ada sisa run sebelumnya (biar fresh)
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)

        print("Menyimpan model ke folder lokal...")
        # Ini akan membuat folder 'model_output' berisi MLmodel, conda.yaml, model.pkl
        mlflow.sklearn.save_model(
            best_model, 
            path=local_model_path,
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        
        print("Mengunggah folder model ke DagsHub...")
        mlflow.log_artifacts(local_model_path, artifact_path="model")
        
        # Folder 'model_output' akan tetap ada di folder.
        print(f"Model tersimpan secara lokal di folder: {local_model_path}")

        # --- 5. Artifacts Lain ---
        mlflow.log_artifact(csv_filename)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        os.remove("confusion_matrix.png")
        
        # Feature Importance
        feature_imp = pd.DataFrame({
            'feature': X.columns, 
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_imp.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")
        os.remove("feature_importance.csv")
        
        print(f"Selesai. Akurasi: {acc}")

if __name__ == "__main__":
    main()
