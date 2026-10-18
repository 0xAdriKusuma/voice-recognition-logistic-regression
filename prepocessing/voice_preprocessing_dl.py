import librosa
import numpy as np

def preprocess_audio_dl(file_path, target_sr=22050, fixed_duration=3.0):
    try:
        #resamoling
        y, sr = librosa.load(file_path, sr=target_sr)
        
        #trimming
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        #normalisasi
        if np.max(np.abs(y_trimmed)) > 0:
            y_normalized = y_trimmed / np.max(np.abs(y_trimmed))
        else:
            y_normalized = y_trimmed
            
        #padding/truncating
        target_length = int(target_sr * fixed_duration)
        
        if len(y_normalized) > target_length:
            y_final = y_normalized[:target_length]
        else:
            pad_length = target_length - len(y_normalized)
            y_final = np.pad(y_normalized, (0, pad_length), mode='constant')
            
        return y_final, sr
    
    except Exception as e:
        print(f"Error {file_path}: {e}")
        return None, None
