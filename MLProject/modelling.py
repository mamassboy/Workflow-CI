import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys

# Force local tracking URI
mlflow.set_tracking_uri("file:./mlruns")
print(f"✓ MLflow tracking: {mlflow.get_tracking_uri()}")

def load_data():
    """Load preprocessed dataset"""
    possible_paths = [
        "telco_churn_preprocessed.csv",
        "./telco_churn_preprocessed.csv",
        os.path.join(os.getcwd(), "telco_churn_preprocessed.csv"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✓ Found dataset at: {path}")
            return pd.read_csv(path)
    
    raise FileNotFoundError(f"Dataset not found. Searched: {possible_paths}")

def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

def plot_feature_importance(model, features, save_path="feature_importance.png"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    plt.figure(figsize=(12, 6))
    plt.title("Top 15 Features")
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [features[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

def plot_roc_curve(y_true, y_proba, save_path="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'darkorange', lw=2, label=f'AUC={auc:.4f}')
    plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

def train():
    """Main training function"""
    
    print("="*60)
    print("CI/CD Pipeline - Model Training")
    print("="*60)
    
    # Load
    print("\n[1/4] Loading data...")
    df = load_data()
    print(f"✓ Data: {df.shape}")
    
    # Prepare
    print("\n[2/4] Preparing...")
    target = df.columns[-1]
    X = df.drop(target, axis=1)
    y = df[target]
    features = X.columns.tolist()
    print(f"✓ Features: {len(features)}")
    
    # Split
    print("\n[3/4] Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Train
    print("\n[4/4] Training...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
    )
    
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    # Log params (avoid conflicts)
    try:
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
    except Exception as e:
        print(f"Warning: Could not log some params: {e}")
    
    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)
    mlflow.log_metric("cv_score", grid.best_score_)
    
    # Log model only if not in CI environment
    is_ci = os.getenv('CI') == 'true' or os.getenv('GITHUB_ACTIONS') == 'true'
    
    if not is_ci:
        try:
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name=None
            )
            print("✓ Model logged")
        except Exception as e:
            print(f"Warning: Model logging skipped: {e}")
    else:
        print("ℹ CI environment detected - skipping model logging")
        # Save model as pickle artifact instead
        try:
            import pickle
            model_path = "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Model saved as {model_path}")
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
    
    # Log artifacts (plots only for CI - no file artifacts due to permission issues)
    try:
        cm_path = plot_confusion_matrix(y_test, y_pred)
        fi_path = plot_feature_importance(model, features)
        roc_path = plot_roc_curve(y_test, y_proba)
        
        print(f"✓ Plots created: {cm_path}, {fi_path}, {roc_path}")
        
        # Only log to MLflow if not in CI (to avoid path issues)
        if not is_ci:
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(fi_path)
            mlflow.log_artifact(roc_path)
            print("✓ Plots logged to MLflow")
        
        # Classification report (as dict, not file)
        report = classification_report(y_test, y_pred, 
                                      target_names=['No Churn', 'Churn'],
                                      output_dict=True)
        report_path = "classification_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"✓ Report saved: {report_path}")
        
        if not is_ci:
            mlflow.log_artifact(report_path)
            
    except Exception as e:
        print(f"Warning: Some artifacts could not be created: {e}")
    
    # Tags
    try:
        mlflow.set_tag("model", "RandomForest")
        mlflow.set_tag("pipeline", "CI/CD")
        mlflow.set_tag("platform", sys.platform)
    except Exception as e:
        print(f"Warning: Could not set tags: {e}")
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print("="*60)
    print("✅ Training Completed!")
    print("="*60)

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)