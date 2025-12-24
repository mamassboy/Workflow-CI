# Workflow-CI: MLflow CI/CD Pipeline

Repository ini berisi workflow CI/CD untuk training model Telco Customer Churn menggunakan MLflow dan GitHub Actions.

## 📁 Struktur Repository

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci-workflow.yml       # GitHub Actions workflow
├── MLProject/
│   ├── MLproject                 # MLflow project config
│   ├── conda.yaml                # Dependencies
│   ├── modelling.py              # Training script
│   └── telco_churn_preprocessed.csv  # Dataset
└── README.md
```

## 🚀 Cara Kerja

1. **Push ke branch main** atau **manual trigger** akan memicu workflow
2. GitHub Actions akan:
   - Setup Python 3.12
   - Install dependencies
   - Menjalankan MLflow Project
   - Training model dengan hyperparameter tuning
   - Upload artifacts ke GitHub

## 📊 Model Details

- **Dataset**: Telco Customer Churn
- **Algorithm**: Random Forest Classifier
- **Tuning**: GridSearchCV
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC AUC

## 🔧 Local Testing

Untuk testing lokal:

```bash
cd MLProject
mlflow run . --env-manager=local
```

## 📈 Artifacts

Setiap training menghasilkan:
- Model (RandomForest)
- Confusion Matrix
- Feature Importance
- ROC Curve
- Classification Report
- CV Results

## 👤 Author

Dimas Arya Arjuna - Dicoding Submission

## 📝 License

Educational purpose only