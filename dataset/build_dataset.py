import os
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm 
from prepocessing.voice_preprocessing_ml import preprocess_audio_ml
from prepocessing.voice_preprocessing_dl import preprocess_audio_dl
from features.feature_extractor_ml import extract_features_ml
from features.feature_extractor_dl import extract_features_dl

DATASET_PATH = "dataset"
# output file csv ML
OUTPUT_CSV = "dataset_fitur_audio.csv"

OUTPUT_X_DL = "x_spectrogram_data.npy"
OUTPUT_Y_DL = "y_gender_labels.npy"


def process_single_file(file_info):
    file_path = file_info['path']
    file_name = file_info['filename']
    label = file_info['label']

    features = {
        'ml': None,
        'dl_x': None,
        'dl_y': None
    }
    
    #panggil fungsi preprocessing
    y_ml, sr_ml = preprocess_audio_ml(file_path)

    if y_ml is not None:
        #panggil fungsi ekstraksi fitur
        features = extract_features_ml(y_ml, sr_ml)
        features['filename'] = file_name
        features['label'] = label
        
    y_dl, sr_dl = preprocess_audio_dl(file_path)
    if y_dl is not None:
        spectrogram = extract_features_dl(y_dl, sr_dl)
        bin_label = 0 if label == "male" else 1
        features['dl_x'] = spectrogram
        features['dl_y'] = bin_label

        
        return features
        
    return None

def main():
    print("membuat tabel")
    
    all_tasks = []
    for label in ["male", "female"]:
        folder_path = os.path.join(DATASET_PATH, label)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.wav') or f.endswith('.mp3')]
            for f in files:
                all_tasks.append({
                    'path': os.path.join(folder_path, f),
                    'filename': f,
                    'label': label
                })
                
        
    print(f"Total file: {len(all_tasks)}")
    
    all_ml_data = []
    x_dl_list = []
    y_dl_list = []
    
    #multithread
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single_file, all_tasks), total=len(all_tasks)))
        
    #hasil
    for res in results:
        if res is not None:
            ml_dict = res.copy()
            
            if 'dl_x' in ml_dict:
                x_dl_list.append(ml_dict.pop('dl_x'))
            if 'dl_y' in ml_dict:
                y_dl_list.append(ml_dict.pop('dl_y'))
                
            all_ml_data.append(ml_dict)
            
    print("done")
    
    #create tabel CSV (Untuk ML)
    if len(all_ml_data) > 0:
        df = pd.DataFrame(all_ml_data)
        
        #pindahkan kolom nama file dan label
        kolom_lain = [c for c in df.columns if c not in ['filename', 'label']]
        df = df[['filename', 'label'] + kolom_lain]
        
        #simpan CSV
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Tabel ML :  {OUTPUT_CSV}")
        
        #simpan NPY (Untuk DL / CNN)
        if len(x_dl_list) > 0:
            np.save(OUTPUT_X_DL, np.array(x_dl_list))
            np.save(OUTPUT_Y_DL, np.array(y_dl_list))
            print(f"Data DL X:  {OUTPUT_X_DL}")
            print(f"Data DL Y:  {OUTPUT_Y_DL}")
    else:
        print("Gagal!")

if __name__ == "__main__":
    main()