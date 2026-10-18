import librosa
import numpy as np

def preprocess_audio_ml(file_path, target_sr=22050):
    try:
        # resampling
        y, sr = librosa.load(file_path, sr=target_sr)
        
        # trimming
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # normalisasi
        if np.max(np.abs(y_trimmed)) > 0:
            y_normalized = y_trimmed / np.max(np.abs(y_trimmed))
        else:
            y_normalized = y_trimmed
            
        return y_normalized, sr
    
    except Exception as e:
        print(f"Error {file_path}: {e}")
        return None, None
