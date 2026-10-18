import os
import librosa
import numpy as np

def extract_features_dl(y, sr):
    features = {}
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_final = np.expand_dims(mel_spec_db, axis=-1)

 
    features['spectrogram_matrix'] = mel_spec_final
    return features