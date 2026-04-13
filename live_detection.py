import sqlite3, threading, queue, datetime, pickle, warnings
import torch, torch.nn as nn, numpy as np
from scapy.all import sniff, IP
from collections import deque

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATABASE WORKER ---
db_queue = queue.Queue()
def database_worker():
    conn = sqlite3.connect("security_logs.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS network_events (id INTEGER PRIMARY KEY, timestamp TEXT, label TEXT, error REAL, threshold REAL, probability REAL)')
    conn.commit()
    while True:
        item = db_queue.get()
        if item is None: break
        cursor.execute('INSERT INTO network_events (timestamp, label, error, threshold, probability) VALUES (?,?,?,?,?)', item)
        conn.commit()
threading.Thread(target=database_worker, daemon=True).start()

# --- MODEL DEFINITIONS ---
class HybridModel(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(16))
        self.fc = nn.Linear(64 * 16, 64)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2)
        self.latent = nn.Linear(64, 8)
        self.decoder = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 17))
    def encode(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x).flatten(1)
        x = self.fc(x)
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        return self.latent(x)
    def forward(self, x): return self.decoder(self.encode(x))

# --- LOAD ---
model = HybridModel(17).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
clf = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
clf.load_state_dict(torch.load("classifier.pth", map_location=device))
clf.eval()
scaler = pickle.load(open("scaler.pkl", "rb"))

recent_errors = deque(maxlen=500)
CALIBRATION_LIMIT = 50

def process_packet(pkt):
    if not pkt.haslayer(IP):
        return # Skip non-IP traffic like ARP/STP

    try:
        # 1. Extract Features
        size = float(len(pkt))
        time_ref = float(pkt.time % 1000)
        tcp = 1.0 if pkt.haslayer("TCP") else 0.0
        udp = 1.0 if pkt.haslayer("UDP") else 0.0
        icmp = 1.0 if pkt.haslayer("ICMP") else 0.0
        payload = float(len(pkt.payload))
        ttl = float(pkt[IP].ttl)

        feats = [size, time_ref, tcp, udp, icmp, payload, ttl]
        while len(feats) < 17:
            feats.append(0.0)

        # 2. Scale & Predict
        sample = scaler.transform(np.array(feats).reshape(1, -1))
        x = torch.tensor(sample, dtype=torch.float32).to(device)

        with torch.no_grad():
            latent = model.encode(x)
            recon = model(x)
            err = ((recon - x)**2).mean().item()
            prob = torch.sigmoid(clf(latent)).item()

        # 3. Dynamic Thresholding
        recent_errors.append(err)
        if len(recent_errors) < CALIBRATION_LIMIT:
            print(f"⏳ CALIBRATING ({len(recent_errors)}/{CALIBRATION_LIMIT}) | Error: {err:.4f}")
            return

        avg = np.mean(recent_errors)
        std = np.std(recent_errors)
        thresh = avg + (3 * std)

        # 4. Decision
        label = "🔴 ATTACK" if (err > thresh or prob > 0.5) else "🟢 NORMAL"
        
        print(f"{label} | Error: {err:.4f} | Thresh: {thresh:.4f} | Prob: {prob:.4f}")
        
        # 5. Log
        db_queue.put((datetime.datetime.now().strftime("%H:%M:%S"), label, err, thresh, prob))

    except Exception as e:
        pass

print(f"🛡️  IDS Started on {device}. Monitoring live IP traffic...")
sniff(prn=process_packet, store=False)