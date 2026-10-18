import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from models.DL.CNN_model import get_cnn_model

def run_cnn_training(x_path="x_spectrogram_data.npy", y_path="y_gender_labels.npy"):
    print("\n--- CNN SPECTROGRAM TRAINING (PyTorch) ---")
    
    X_raw = np.load(x_path, allow_pickle=True)
    y_raw = np.load(y_path, allow_pickle=True)
    
    if isinstance(X_raw[0], dict):
        X = np.array([item['spectrogram_matrix'] for item in X_raw])
    else:
        X = X_raw
        
    y = y_raw.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Kosongkan RAM
    del X_raw, y_raw, X, y
    
    X_train_t = torch.tensor(X_train).float().permute(0, 3, 1, 2)
    X_test_t = torch.tensor(X_test).float().permute(0, 3, 1, 2)
    y_train_t = torch.tensor(y_train).float().unsqueeze(1)
    y_test_t = torch.tensor(y_test).float().unsqueeze(1)
    
    del X_train, X_test, y_train, y_test
    import gc
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = get_cnn_model() 
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 1
    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for val_inputs, val_labels in test_loader:
                val_outputs = model(val_inputs)
                v_loss = criterion(val_outputs, val_labels)
                val_running_loss += v_loss.item()
                
                val_preds = (val_outputs > 0.5).float()
                val_correct += (val_preds == val_labels).sum().item()
                val_total += val_labels.size(0)
                
        val_loss = val_running_loss / len(test_loader)
        val_acc = val_correct / val_total
        
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        gc.collect()
        
    print(f"\nAkurasi Akhir CNN pada Data Test: {history['val_accuracy'][-1]:.4f}")
    
    return {'CNN': history['val_accuracy'][-1], 'history': history, 'model': model}