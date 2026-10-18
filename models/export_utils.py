import os
import joblib
import torch

def save_ml_model(model, filename, save_dir="exported_models"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    joblib.dump(model, filepath)
    print(f"Model ML berhasil disimpan di: {filepath}")

def save_dl_model(model, filename, save_dir="exported_models"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Bobot model CNN berhasil disimpan di: {filepath}")
