import os
import joblib
import torch
import librosa
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys

# Tambahkan direktori root agar bisa import module lokal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.feature_extractor_ml import extract_features_ml
from features.feature_extractor_dl import extract_features_dl
from models.DL.CNN_model import get_cnn_model

app = Flask(__name__)
CORS(app)

# Tentukan path models dan dataset
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'exported_models')
CSV_PATH = os.path.join(BASE_DIR, 'dataset_fitur_audio.csv')

# Load CSV Headers untuk menjaga urutan fitur yang tepat
try:
    df_dummy = pd.read_csv(CSV_PATH, nrows=0)
    ALL_COLUMNS = df_dummy.columns.tolist()
except Exception as e:
    print(f"Error loading CSV for headers: {e}")
    ALL_COLUMNS = []

# Load models and scalers
models_cache = {}

def load_all_models():
    print("Loading models into memory...")
    try:
        # Load ML Models
        for exp in ['exp1_mfcc', 'exp2_pitch', 'exp3_hybrid']:
            for model_name in ['rf', 'svm', 'lr']:
                filename = f"{model_name}_{exp}.joblib"
                path = os.path.join(MODELS_DIR, filename)
                if os.path.exists(path):
                    models_cache[f"{model_name}_{exp}"] = joblib.load(path)
            
            # Load scalers
            scaler_name = f"scaler_{exp}.joblib"
            scaler_path = os.path.join(MODELS_DIR, scaler_name)
            if os.path.exists(scaler_path):
                models_cache[f"scaler_{exp}"] = joblib.load(scaler_path)
                
        # Load DL Model (CNN)
        cnn_path = os.path.join(MODELS_DIR, "cnn_spectrogram_weights.pth")
        if os.path.exists(cnn_path):
            cnn = get_cnn_model()
            cnn.load_state_dict(torch.load(cnn_path, map_location=torch.device('cpu')))
            cnn.eval()
            models_cache['cnn'] = cnn
            
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Failed to load models: {e}")

# Panggil fungsi load saat aplikasi start
load_all_models()

def get_ordered_features(extracted_features_dict, scenario):
    """Memfilter fitur berdasarkan skenario dengan urutan persis seperti saat training"""
    if scenario == 'exp1':
        prefix = ('mfcc_', 'centroid_', 'bandwidth_', 'rolloff_', 'rms_', 'zcr_')
    elif scenario == 'exp2':
        prefix = ('pitch_', 'centroid_', 'bandwidth_', 'rolloff_', 'rms_', 'zcr_')
    elif scenario == 'exp3':
        prefix = ('mfcc_', 'pitch_', 'centroid_', 'bandwidth_', 'rolloff_', 'rms_', 'zcr_')
    else:
        return []
        
    ordered_features = []
    # Loop melalui kolom CSV asli untuk menjaga urutan
    for col in ALL_COLUMNS:
        if col.startswith(prefix) and col in extracted_features_dict:
            ordered_features.append(extracted_features_dict[col])
            
    return np.array(ordered_features).reshape(1, -1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'Tidak ada file audio yang diunggah'}), 400
        
    audio_file = request.files['audio']
    model_type = request.form.get('model', 'rf')
    scenario = request.form.get('scenario', 'exp3')
    
    try:
        # Save to temp file first because librosa can't always read direct Flask FileStorage
        import tempfile
        import uuid
        temp_filename = f"temp_{uuid.uuid4().hex}.wav"
        audio_file.save(temp_filename)
        
        # Load audio (mono, 22050 Hz)
        y, sr = librosa.load(temp_filename, sr=22050)
        
        # Hapus file temp
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        # Preprocessing Dasar (Sama persis dengan saat training)
        # 1. Hapus keheningan (silence) di awal dan akhir rekaman
        y, _ = librosa.effects.trim(y, top_db=20)
            
        # 2. Potong maksimum 5 detik (5 * 22050) untuk ML, jika kepanjangan
        max_samples = 5 * 22050
        if len(y) > max_samples:
            y = y[:max_samples]
            
        prediction = None
        confidence = None
            
        if model_type == 'cnn':
            if 'cnn' not in models_cache:
                return jsonify({'error': 'Model CNN belum di-export. Silakan export dulu di Jupyter Notebook.'}), 500
                
            # Pad or trim to exactly 3 seconds for CNN (as per voice_preprocessing_dl)
            target_length = 66150 # 3 seconds * 22050
            if len(y) > target_length:
                y_cnn = y[:target_length]
            else:
                y_cnn = np.pad(y, (0, target_length - len(y)))
                
            features = extract_features_dl(y_cnn, sr)
            spec_matrix = features['spectrogram_matrix']
            
            # Bentuk tensor: (batch=1, channel=1, height=128, width=130)
            tensor_input = torch.tensor(spec_matrix).float().unsqueeze(0).permute(0, 3, 1, 2)
            
            with torch.no_grad():
                output = models_cache['cnn'](tensor_input)
                prob = output.item()
                
            prediction = 'Female' if prob > 0.5 else 'Male'
            confidence = prob if prob > 0.5 else 1 - prob
            
        else: # ML Models (RF, SVM, LR)
            # Konversi scenario code
            scenario_map = {'1': 'exp1', '2': 'exp2', '3': 'exp3'}
            if scenario in scenario_map:
                scenario = scenario_map[scenario]
                
            # Cek ketersediaan model dan scaler
            model_key = f"{model_type}_{scenario}"
            scaler_key = f"scaler_{scenario}"
            
            # Coba tangani format nama jika user menggunakan nama hybrid
            if scenario == 'exp1': model_key = f"{model_type}_exp1_mfcc"
            elif scenario == 'exp2': model_key = f"{model_type}_exp2_pitch"
            elif scenario == 'exp3': model_key = f"{model_type}_exp3_hybrid"
            
            if scenario == 'exp1': scaler_key = f"scaler_exp1_mfcc"
            elif scenario == 'exp2': scaler_key = f"scaler_exp2_pitch"
            elif scenario == 'exp3': scaler_key = f"scaler_exp3_hybrid"
            
            if model_key not in models_cache or scaler_key not in models_cache:
                return jsonify({'error': f'Model {model_key} atau Scaler {scaler_key} belum di-export.'}), 500
                
            model = models_cache[model_key]
            scaler = models_cache[scaler_key]
            
            # Ekstraksi dan penyusunan fitur
            raw_features = extract_features_ml(y, sr)
            scenario_code = scenario.split('_')[0] # exp1, exp2, exp3
            X_input = get_ordered_features(raw_features, scenario_code)
            
            if X_input.shape[1] == 0:
                return jsonify({'error': 'Gagal mengekstrak fitur. Pastikan CSV dataset ada di folder utama.'}), 500
                
            X_scaled = scaler.transform(X_input)
            
            pred = model.predict(X_scaled)[0]
            prediction = 'Female' if pred == 1 else 'Male'
            
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_scaled)[0]
                confidence = probs[pred]
            else:
                confidence = 1.0 # SVM biasanya tidak memiliki probabilitas kecuali probability=True diset
                
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': f"{confidence * 100:.2f}%",
            'model_used': model_type.upper(),
            'scenario': scenario
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Pastikan folder templates ada
    os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)
    app.run(debug=True, port=5000)
