from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
import torch.nn as nn
import numpy as np
import pickle

app = FastAPI()
device = "cpu"
FEATURE_COUNT = 17 
THRESHOLD = 35.0  

# =========================
# MODEL ARCHITECTURE
# =========================
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

# =========================
# LOAD FILES
# =========================
model = HybridModel(FEATURE_COUNT).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

clf = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
clf.load_state_dict(torch.load("classifier.pth", map_location=device))
clf.eval()

scaler = pickle.load(open("scaler.pkl", "rb"))

class InputData(BaseModel):
    sample: List[float]

@app.post("/predict")
def detect(data: InputData):
    sample = np.array(data.sample).reshape(1, -1)
    sample = scaler.transform(sample)
    sample = np.clip(sample, -20, 20)
    x = torch.tensor(sample, dtype=torch.float32).to(device)

    with torch.no_grad():
        recon = model(x)
        err = ((recon - x) ** 2).mean().item()
        z = model.encode(x)
        prob = torch.sigmoid(clf(z)).item()

    result = "ATTACK" if (err > THRESHOLD or prob > 0.5) else "NORMAL"
    return {"prediction": result, "error": round(err, 4), "probability": round(prob, 4)}