import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os

# --- 1. DEVICE INITIALIZATION ---
def get_train_device():
    if torch.cuda.is_available():
        try:
            # Smoke test to ensure GPU isn't "zombie" or locked
            t = torch.zeros(1).cuda()
            print(f"🚀 GPU Validated: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        except Exception as e:
            print(f"⚠️ GPU Issue: {e}. Defaulting to CPU for stability.")
    return torch.device("cpu")

device = get_train_device()

# --- 2. LOAD & CLEAN DATA ---
print("📊 Loading and preparing dataset...")
try:
    df = pd.read_csv("Network_dataset_1.csv", low_memory=False)
    df['type'] = df['type'].astype(str).str.lower()
    
    normal_df = df[df['type'] == 'normal']
    attack_df = df[df['type'] == 'scanning']
    
    # Strictly enforce 17 features
    numeric_cols = normal_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    features_to_use = numeric_cols[:17] 
    
    normal_data = normal_df[features_to_use].fillna(0).values.astype(np.float32)
    attack_data = attack_df[features_to_use].fillna(0).values.astype(np.float32)
except Exception as e:
    print(f"❌ Data Loading Error: {e}")
    exit()

# --- 3. SCALING ---
scaler = RobustScaler()
normal_np = np.clip(scaler.fit_transform(normal_data), -20, 20)
attack_np = np.clip(scaler.transform(attack_data), -20, 20)
pickle.dump(scaler, open("scaler.pkl", "wb"))

# --- 4. MODEL ARCHITECTURE ---
class HybridModel(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(16)
        )
        self.fc = nn.Linear(64 * 16, 64)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2)
        self.latent = nn.Linear(64, 8)
        self.decoder = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, f))

    def encode(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x).flatten(1)
        x = self.fc(x)
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        return self.latent(x)

    def forward(self, x):
        return self.decoder(self.encode(x))

# --- 5. TRAINING WITH FAULT TOLERANCE ---
model = HybridModel(17).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
# Scaler for Mixed Precision (prevents crashes on older GPUs)
grad_scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

print(f"🏋️ Training Hybrid CNN-Transformer on {device}...")

loader = DataLoader(TensorDataset(torch.tensor(normal_np)), batch_size=1024, shuffle=True)

try:
    for epoch in range(10):
        model.train()
        epoch_loss = 0
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()
            
            # Using 'with autocast' for GPU efficiency
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                recon = model(x + 0.01 * torch.randn_like(x))
                loss = loss_fn(recon, x)
            
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(loader):.6f}")

    # --- 6. TRAIN CLASSIFIER ---
    model.eval()
    with torch.no_grad():
        z_norm = model.encode(torch.tensor(normal_np).to(device))
        z_att = model.encode(torch.tensor(attack_np).to(device))

    N = min(25000, len(z_norm), len(z_att))
    X_clf = torch.cat([z_norm[:N], z_att[:N]])
    y_clf = torch.cat([torch.zeros(N).to(device), torch.ones(N).to(device)])

    classifier = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
    optimizer2 = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    loss_fn2 = nn.BCEWithLogitsLoss()

    loader2 = DataLoader(TensorDataset(X_clf, y_clf), batch_size=512, shuffle=True)
    for epoch in range(5):
        for xb, yb in loader2:
            optimizer2.zero_grad()
            loss_fn2(classifier(xb).squeeze(), yb).backward()
            optimizer2.step()

    # SAVE EVERYTHING
    torch.save(model.state_dict(), "model.pth")
    torch.save(classifier.state_dict(), "classifier.pth")
    print("✅ Success! Model weights saved to model.pth and classifier.pth")

except RuntimeError as e:
    if "out of memory" in str(e):
        print("🚨 GPU Memory Full! Try reducing the batch_size in the script.")
    else:
        print(f"❌ Training Crash: {e}")