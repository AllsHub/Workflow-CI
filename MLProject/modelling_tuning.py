import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn

# Menggunakan Environment Variable
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/AllsHub/Submission_MLOps_aldomp7.mlflow"
# Username & Password akan diambil otomatis dari Secrets GitHub Action

def main():
    mlflow.set_experiment("Breast Cancer Classification - CI Workflow")
    
    with mlflow.start_run():
        
        csv_filename = 'breastcancer_preprocessing.csv' 
        
        if not os.path.exists(csv_filename):
            print("Error: File CSV tidak ditemukan.")
            return
        
        print(f"Membaca data dari file: {csv_filename}")
        df = pd.read_csv(csv_filename)
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf = RandomForestClassifier(random_state=42)
        params = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10]
        }
        
        clf = GridSearchCV(rf, params, cv=3, scoring='accuracy')
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_
        
        y_pred = best_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        mlflow.log_params(clf.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        mlflow.sklearn.log_model(best_model, "model")
        
        print(f"Training Selesai. Akurasi: {acc}")

if __name__ == "__main__":
    main()