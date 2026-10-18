import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from models.ML.ml_models import get_logistic_model, get_svm_model, get_rf_model

def run_experiment_3(csv_path="dataset_fitur_audio.csv"):
    df = pd.read_csv(csv_path)
    
    fitur_skenario_3 = [c for c in df.columns if c.startswith(('mfcc_', 'pitch_', 'centroid_', 'bandwidth_', 'rolloff_', 'rms_', 'zcr_'))]
    X = df[fitur_skenario_3]
    y = df['label'].map({'male': 0, 'female': 1})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    dict_models = {
        'Logistic Regression': get_logistic_model(),
        'SVM': get_svm_model(),
        'Random Forest': get_rf_model()
    }
    
    results = {}
    for name, model in dict_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds)
        
        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': cm,
            'model': model
        }
        
        print(f"\n--- Evaluasi {name} ---")
        print(f"Akurasi          : {acc:.4f}")
        print(f"Precision        : {prec:.4f}")
        print(f"Recall           : {rec:.4f}")
        print(f"F1-Score         : {f1:.4f}")
        print(f"Confusion Matrix :\n{cm}")
        
    results['scaler'] = scaler
    return results