import librosa
import numpy as np

def extract_features_ml(y, sr):
    features = {}
    
    #ekstraksi fitur pitch
    try:
        pitch = librosa.yin(y, fmin=50, fmax=500)
    except:
        pitch = np.zeros(1) 
        
    #ekstraksi fitur MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    #fitur domain waktu
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    mean_amp = np.abs(y)
    
    #fitur domain frekuensi
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    
    #feature statistik
    
    def hitung_statistik(nama_fitur, data_array):
        features[f"{nama_fitur}_mean"] = np.mean(data_array)
        features[f"{nama_fitur}_std"] = np.std(data_array)
        features[f"{nama_fitur}_min"] = np.min(data_array)
        features[f"{nama_fitur}_max"] = np.max(data_array)

    # hitung
    hitung_statistik("pitch", pitch)
    hitung_statistik("rms", rms)
    hitung_statistik("zcr", zcr)
    hitung_statistik("mean_amp", mean_amp)
    hitung_statistik("centroid", spectral_centroid)
    hitung_statistik("bandwidth", spectral_bandwidth)
    hitung_statistik("rolloff", spectral_rolloff)
    
    for i in range(13):
        hitung_statistik(f"mfcc_{i+1}", mfcc[i])
        
    return features

