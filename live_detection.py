import os
import pickle
import sqlite3
import threading
import queue
import time
import datetime as dt
import ipaddress

import numpy as np
import torch
import torch.nn as nn

from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP, ICMP


# ============================================================
# HYBRID LIVE IDS
# 1 = Network
# 2 = IoT
#
# Both modes:
#   - load their own model/scaler
#   - perform a 2-minute startup calibration
#   - calculate a 99th-percentile runtime threshold
#   - freeze that threshold
#   - then begin live detection
#
# No continuous threshold adaptation.
# ============================================================


# ============================================================
# GENERAL CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CALIBRATION_SECONDS = 120
CALIBRATION_PERCENTILE = 99.0
MIN_CALIBRATION_SAMPLES = 30

FLOW_TIMEOUT = 3

ATTACK_PROBABILITY_THRESHOLD = 0.70

DB_NAME = "ids_logs.db"

CAPTURE_INTERFACES = [
    r"\Device\NPF_Loopback",
    r"\Device\NPF_{97F6B005-D45F-4C53-875D-E9B7522980F6}"
]




# ============================================================
# AUDIO ALARM
# ============================================================
# Classifier-only detections are treated as SUSPICIOUS rather
# than full attacks. The autoencoder is used as the second
# signal. Both signals agreeing produces the strongest alarm.
# ============================================================
try:
    import winsound
except ImportError:
    winsound = None

ALARM_COOLDOWN_SECONDS = 2.0
_last_alarm_time = 0.0
_alarm_lock = threading.Lock()


def _beep_worker(pattern):
    if winsound is None:
        return

    try:
        for frequency, duration, pause in pattern:
            winsound.Beep(frequency, duration)
            if pause > 0:
                time.sleep(pause)
    except Exception:
        pass


def trigger_alarm(level):
    """Trigger a non-blocking Windows alarm with a short cooldown."""
    global _last_alarm_time

    if winsound is None or level <= 0:
        return

    now = time.time()
    with _alarm_lock:
        if now - _last_alarm_time < ALARM_COOLDOWN_SECONDS:
            return
        _last_alarm_time = now

    patterns = {
        # Classifier-only: suspicious, deliberately mild.
        1: [(700, 180, 0.0)],
        # Reconstruction-only: anomaly/possible zero-day.
        2: [(850, 180, 0.12), (850, 180, 0.0)],
        # Both signals agree: attack severity increases with probability.
        3: [(1000, 180, 0.12), (1000, 180, 0.12), (1000, 180, 0.0)],
        4: [(1200, 180, 0.10), (1200, 180, 0.10), (1200, 180, 0.10), (1200, 180, 0.0)],
        5: [(1500, 220, 0.08), (1500, 220, 0.08), (1500, 220, 0.08), (1500, 220, 0.08), (1500, 220, 0.0)],
    }

    threading.Thread(
        target=_beep_worker,
        args=(patterns.get(level, patterns[1]),),
        daemon=True
    ).start()


def classify_detection(error, threshold, probability):
    """
    Hybrid decision:
      NORMAL      : neither branch fires.
      SUSPICIOUS  : classifier alone fires.
      ATTACK      : reconstruction fires; especially strong when
                    classifier and reconstruction agree.

    Audible levels:
      0 = none
      1 = classifier-only suspicious
      2 = reconstruction-only anomaly
      3-5 = both branches agree, scaled by probability
    """
    reconstruction_attack = error > threshold
    classifier_attack = probability >= ATTACK_PROBABILITY_THRESHOLD

    if reconstruction_attack and classifier_attack:
        if probability >= 0.95:
            return "ATTACK", "classifier + reconstruction", 5
        if probability >= 0.85:
            return "ATTACK", "classifier + reconstruction", 4
        return "ATTACK", "classifier + reconstruction", 3

    if reconstruction_attack:
        return "ATTACK", "reconstruction anomaly", 2

    if classifier_attack:
        return "SUSPICIOUS", "classifier only", 1

    return "NORMAL", "normal", 0


# ============================================================
# MODE SELECTION
# ============================================================

def choose_mode():

    while True:

        print()
        print("=" * 70)
        print("             HYBRID CNN + TRANSFORMER IDS")
        print("=" * 70)
        print()
        print("Choose detection mode:")
        print()
        print("  1. Network / Wi-Fi")
        print("  2. Mobile Data / Hotspot")
        print("  3. IoT")
        print("  4. Exit")
        print()

        choice = input(
            "Enter choice [1/2/3/4]: "
        ).strip()

        if choice in ("1", "2", "3", "4"):
            return choice

        print()
        print("Invalid choice. Please enter 1, 2, 3, or 4.")


# ============================================================
# DATABASE
# ============================================================

db_queue = queue.Queue()


def database_worker():

    conn = sqlite3.connect(
        DB_NAME,
        check_same_thread=False
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS logs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            mode TEXT,
            src_ip TEXT,
            dst_ip TEXT,
            prediction TEXT,
            reconstruction_error REAL,
            threshold REAL,
            probability REAL
        )
        """
    )

    conn.commit()

    while True:

        item = db_queue.get()

        if item is None:

            db_queue.task_done()
            conn.close()
            break

        try:

            cursor.execute(
                """
                INSERT INTO logs(
                    timestamp,
                    mode,
                    src_ip,
                    dst_ip,
                    prediction,
                    reconstruction_error,
                    threshold,
                    probability
                )
                VALUES(?,?,?,?,?,?,?,?)
                """,
                item
            )

            conn.commit()

        except Exception as e:

            print(
                "[DATABASE ERROR]",
                repr(e)
            )

        finally:

            db_queue.task_done()


threading.Thread(
    target=database_worker,
    daemon=True
).start()


# ============================================================
# COMMON HELPERS
# ============================================================

def is_multicast_or_broadcast(ip):

    try:

        address = ipaddress.ip_address(ip)

        if address.is_multicast:
            return True

        if ip == "255.255.255.255":
            return True

        if ip == "0.0.0.0":
            return True

        return False

    except Exception:

        return False


def safe_mean(values):

    return float(np.mean(values)) if values else 0.0


def safe_std(values):

    return float(np.std(values)) if len(values) >= 2 else 0.0


def safe_min(values):

    return float(np.min(values)) if values else 0.0


def safe_max(values):

    return float(np.max(values)) if values else 0.0


def inter_arrival_times(values):

    if len(values) < 2:
        return []

    values = sorted(values)

    return [
        values[i] - values[i - 1]
        for i in range(1, len(values))
    ]


# ============================================================
# NETWORK MODEL
# Exact architecture used by the new 30-feature training
# ============================================================

class NetworkModel(nn.Module):

    def __init__(self, feature_count=30, latent_size=16):
        super().__init__()

        # EXACT architecture used by the trained network_model.pth
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(16)
        )

        self.fc = nn.Linear(64 * 16, 64)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=2
        )

        self.latent = nn.Linear(64, latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, feature_count)
        )

    def encode(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        return self.latent(x)

    def forward(self, x):
        return self.decoder(self.encode(x))


# ============================================================
# NETWORK CLASSIFIER
# ============================================================

network_classifier = nn.Sequential(
    nn.Linear(16, 64),
    nn.ReLU(),
    nn.Dropout(0.20),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)


# ============================================================
# NETWORK FEATURES
# EXACT ORDER USED DURING NEW TRAINING
# ============================================================

NETWORK_FEATURE_COLUMNS = [

    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count"
]


# ============================================================
# NETWORK DETECTOR
# ============================================================

class NetworkDetector:

    def __init__(self):

        self.model_dir = "network_model"

        self.model_file = os.path.join(
            self.model_dir,
            "network_model.pth"
        )

        self.classifier_file = os.path.join(
            self.model_dir,
            "network_classifier.pth"
        )

        self.scaler_file = os.path.join(
            self.model_dir,
            "network_scaler.pkl"
        )

        self.threshold_file = os.path.join(
            self.model_dir,
            "network_threshold.pkl"
        )

        self.runtime_threshold_file = os.path.join(
            self.model_dir,
            "network_runtime_threshold.pkl"
        )

        self.model = NetworkModel(30).to(DEVICE)

        self.classifier = network_classifier.to(DEVICE)

        self.flows = {}

        self.calibrating = True

        self.calibration_start = None

        self.calibration_errors = []

        self.threshold = None

        self.load()


    def load(self):

        print()
        print("[NETWORK] Loading model...")

        self.model.load_state_dict(
            torch.load(
                self.model_file,
                map_location=DEVICE
            )
        )

        self.model.eval()

        self.classifier.load_state_dict(
            torch.load(
                self.classifier_file,
                map_location=DEVICE
            )
        )

        self.classifier.eval()

        with open(
            self.scaler_file,
            "rb"
        ) as f:

            self.scaler = pickle.load(f)

        with open(
            self.threshold_file,
            "rb"
        ) as f:

            self.training_threshold = float(
                pickle.load(f)
            )

        self.threshold = self.training_threshold

        print(
            "[NETWORK] Model loaded."
        )

        print(
            "[NETWORK] Training threshold:",
            f"{self.training_threshold:.6f}"
        )


    def flow_key(self, pkt):

        src = pkt[IP].src
        dst = pkt[IP].dst

        proto = pkt[IP].proto

        sport = 0
        dport = 0

        if pkt.haslayer(TCP):

            sport = int(pkt[TCP].sport)
            dport = int(pkt[TCP].dport)

        elif pkt.haslayer(UDP):

            sport = int(pkt[UDP].sport)
            dport = int(pkt[UDP].dport)

        a = (
            src,
            sport,
            dst,
            dport,
            proto
        )

        b = (
            dst,
            dport,
            src,
            sport,
            proto
        )

        return min(a, b)


    def create_flow(self, pkt):

        sport = 0
        dport = 0

        if pkt.haslayer(TCP):

            sport = int(pkt[TCP].sport)
            dport = int(pkt[TCP].dport)

        elif pkt.haslayer(UDP):

            sport = int(pkt[UDP].sport)
            dport = int(pkt[UDP].dport)

        timestamp = float(pkt.time)

        return {

            "start_time": timestamp,
            "last_seen": timestamp,

            "src_ip": pkt[IP].src,
            "dst_ip": pkt[IP].dst,

            "src_port": sport,
            "dst_port": dport,

            "fwd_times": [],
            "bwd_times": [],

            "fwd_lengths": [],
            "bwd_lengths": [],

            "fin": 0,
            "syn": 0,
            "rst": 0
        }


    def update_flow(
        self,
        flow,
        pkt
    ):

        now = float(pkt.time)

        flow["last_seen"] = now

        length = len(pkt)

        is_forward = (
            pkt[IP].src == flow["src_ip"]
        )

        if is_forward:

            flow["fwd_times"].append(now)
            flow["fwd_lengths"].append(length)

        else:

            flow["bwd_times"].append(now)
            flow["bwd_lengths"].append(length)

        if pkt.haslayer(TCP):

            flags = int(pkt[TCP].flags)

            if flags & 0x01:
                flow["fin"] += 1

            if flags & 0x02:
                flow["syn"] += 1

            if flags & 0x04:
                flow["rst"] += 1


    def build_features(self, flow):

        fwd = flow["fwd_lengths"]
        bwd = flow["bwd_lengths"]

        all_lengths = fwd + bwd

        fwd_times = flow["fwd_times"]
        bwd_times = flow["bwd_times"]

        all_times = fwd_times + bwd_times

        duration = max(
            0.0,
            flow["last_seen"] - flow["start_time"]
        )

        total_fwd = len(fwd)
        total_bwd = len(bwd)

        total_fwd_bytes = sum(fwd)
        total_bwd_bytes = sum(bwd)

        total_packets = total_fwd + total_bwd
        total_bytes = total_fwd_bytes + total_bwd_bytes

        all_iat = inter_arrival_times(all_times)
        fwd_iat = inter_arrival_times(fwd_times)
        bwd_iat = inter_arrival_times(bwd_times)

        if duration > 0:

            flow_bytes_sec = total_bytes / duration
            flow_packets_sec = total_packets / duration
            fwd_packets_sec = total_fwd / duration
            bwd_packets_sec = total_bwd / duration

        else:

            flow_bytes_sec = 0.0
            flow_packets_sec = 0.0
            fwd_packets_sec = 0.0
            bwd_packets_sec = 0.0

        features = [

            flow["dst_port"],
            duration,

            total_fwd,
            total_bwd,

            total_fwd_bytes,
            total_bwd_bytes,

            safe_max(fwd),
            safe_mean(fwd),

            safe_max(bwd),
            safe_mean(bwd),

            flow_bytes_sec,
            flow_packets_sec,

            safe_mean(all_iat),
            safe_std(all_iat),
            safe_max(all_iat),
            safe_min(all_iat),

            safe_mean(fwd_iat),
            safe_std(fwd_iat),

            safe_mean(bwd_iat),
            safe_std(bwd_iat),

            fwd_packets_sec,
            bwd_packets_sec,

            safe_min(all_lengths),
            safe_max(all_lengths),
            safe_mean(all_lengths),
            safe_std(all_lengths),

            float(np.var(all_lengths))
            if all_lengths else 0.0,

            flow["fin"],
            flow["syn"],
            flow["rst"]
        ]

        return np.asarray(
            features,
            dtype=np.float32
        ).reshape(1, -1)


    def infer(self, flow):

        sample = self.build_features(flow)

        if sample.shape != (1, 30):

            raise RuntimeError(
                f"[NETWORK] Bad feature shape: {sample.shape}"
            )

        sample = self.scaler.transform(sample)

        sample = np.clip(
            sample,
            -20,
            20
        )

        x = torch.tensor(
            sample,
            dtype=torch.float32,
            device=DEVICE
        )

        with torch.no_grad():

            z = self.model.encode(x)

            reconstruction = self.model.decoder(z)

            error = (
                (reconstruction - x) ** 2
            ).mean().item()

            probability = torch.sigmoid(
                self.classifier(z)
            ).item()

        return error, probability


    def calibrate(self, flow):

        try:

            error, _ = self.infer(flow)

            self.calibration_errors.append(
                error
            )

        except Exception as e:

            print(
                "[NETWORK CALIBRATION ERROR]",
                repr(e)
            )


    def finish_calibration(self):

        count = len(
            self.calibration_errors
        )

        print()
        print("=" * 70)
        print("NETWORK CALIBRATION COMPLETE")
        print("=" * 70)

        print(
            "Samples:",
            count
        )

        if count < MIN_CALIBRATION_SAMPLES:

            print(
                "WARNING: insufficient calibration samples."
            )

            print(
                "Keeping training threshold:",
                f"{self.training_threshold:.6f}"
            )

        else:

            errors = np.asarray(
                self.calibration_errors,
                dtype=np.float64
            )

            mean_error = np.mean(errors)
            median_error = np.median(errors)
            std_error = np.std(errors)

            percentile = np.percentile(
                errors,
                CALIBRATION_PERCENTILE
            )

            sigma = (
                mean_error
                +
                3.0 * std_error
            )

            self.threshold = max(
                percentile,
                sigma
            )

            if (
                not np.isfinite(self.threshold)
                or
                self.threshold <= 0
            ):

                self.threshold = (
                    self.training_threshold
                )

            print(
                f"Mean Error       : {mean_error:.6f}"
            )

            print(
                f"Median Error     : {median_error:.6f}"
            )

            print(
                f"Std Error        : {std_error:.6f}"
            )

            print(
                f"99th Percentile  : {percentile:.6f}"
            )

            print(
                f"Mean + 3*Std     : {sigma:.6f}"
            )

            print(
                f"RUNTIME THRESHOLD: {self.threshold:.6f}"
            )

            try:

                with open(
                    self.runtime_threshold_file,
                    "wb"
                ) as f:

                    pickle.dump(
                        float(self.threshold),
                        f
                    )

            except Exception as e:

                print(
                    "Could not save runtime threshold:",
                    repr(e)
                )

        self.calibrating = False

        print()
        print(
            "Calibration traffic was NOT classified."
        )

        print(
            "Threshold is now FROZEN."
        )

        print(
            "NETWORK LIVE DETECTION STARTING."
        )

        print("=" * 70)


    def detect(self, flow):

        try:

            error, probability = self.infer(flow)

        except Exception as e:

            print(
                "[NETWORK DETECTION ERROR]",
                repr(e)
            )

            return

        prediction, reason, alarm_level = classify_detection(
            error,
            self.threshold,
            probability
        )

        print()
        print(
            f"[NETWORK] {prediction}"
        )

        print(
            f"{flow['src_ip']} -> "
            f"{flow['dst_ip']}"
        )

        print(
            f"Error={error:.6f} | "
            f"Threshold={self.threshold:.6f} | "
            f"Probability={probability:.4f}"
        )

        print(
            f"Reason: {reason}."
        )

        print(
            f"Alarm Level: {alarm_level}"
        )

        if prediction == "SUSPICIOUS":
            print(
                "Status: classifier-only detection; full attack confirmation not reached."
            )
        elif prediction == "ATTACK":
            print(
                "Status: attack/anomaly detected."
            )

        trigger_alarm(alarm_level)

        db_queue.put(
            (
                dt.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                "NETWORK",

                flow["src_ip"],
                flow["dst_ip"],

                prediction,

                float(error),
                float(self.threshold),
                float(probability)
            )
        )


    def process_packet(self, pkt):

        if not pkt.haslayer(IP):
            return

        src = pkt[IP].src
        dst = pkt[IP].dst

        if (
            is_multicast_or_broadcast(src)
            or
            is_multicast_or_broadcast(dst)
        ):
            return

        key = self.flow_key(pkt)

        if key not in self.flows:

            self.flows[key] = (
                self.create_flow(pkt)
            )

        self.update_flow(
            self.flows[key],
            pkt
        )


    def expire_flows(self):

        now = time.time()

        expired = []

        for key, flow in list(
            self.flows.items()
        ):

            if (
                now - flow["last_seen"]
                >=
                FLOW_TIMEOUT
            ):

                if self.calibrating:

                    self.calibrate(flow)

                else:

                    self.detect(flow)

                expired.append(key)

        for key in expired:

            self.flows.pop(
                key,
                None
            )


    def run(self):

        print()
        print("=" * 70)
        print("NETWORK MODE")
        print("=" * 70)

        print(
            "2-minute calibration starting."
        )

        print()

        self.calibrating = True
        self.calibration_start = time.time()
        self.calibration_errors.clear()

        last_report = -1

        try:

            while self.calibrating:

                sniff(
                    iface=CAPTURE_INTERFACES,
                    prn=self.process_packet,
                    store=False,
                    filter="ip",
                    timeout=1
                )

                self.expire_flows()

                elapsed = (
                    time.time()
                    -
                    self.calibration_start
                )

                remaining = max(
                    0,
                    CALIBRATION_SECONDS
                    -
                    int(elapsed)
                )

                report = int(
                    elapsed // 10
                )

                if report != last_report:

                    last_report = report

                    print(
                        f"[NETWORK CALIBRATION] "
                        f"{remaining:03d}s remaining | "
                        f"flows={len(self.calibration_errors)}"
                    )

                if elapsed >= CALIBRATION_SECONDS:

                    self.finish_calibration()

            print()

            while True:

                sniff(
                    iface=CAPTURE_INTERFACES,
                    prn=self.process_packet,
                    store=False,
                    filter="ip",
                    timeout=1
                )

                self.expire_flows()

        except KeyboardInterrupt:

            print(
                "\nNetwork IDS stopped."
            )



# ============================================================
# MOBILE DATA / HOTSPOT MODEL
# ============================================================
# IMPORTANT:
# This is a completely separate model from Network/Wi-Fi and IoT.
#
# Mobile mode uses:
#   - its own 30-feature model
#   - its own classifier
#   - its own scaler
#   - its own threshold
#   - its own runtime threshold file
#
# Expected directory:
#   mobile_model/
#       mobile_model.pth
#       mobile_classifier.pth
#       mobile_scaler.pkl
#       mobile_threshold.pkl
#
# The 30 features use the SAME ordering as the mobile model training.
# ============================================================

class MobileModel(nn.Module):

    def __init__(self, feature_count=30, latent_size=8):

        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(16)
        )

        self.fc = nn.Linear(64 * 16, 64)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=2
        )

        self.latent = nn.Linear(64, latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, feature_count)
        )

    def encode(self, x):

        x = x.unsqueeze(1)

        x = self.cnn(x)

        x = x.flatten(1)

        x = self.fc(x)

        x = self.transformer(
            x.unsqueeze(1)
        ).squeeze(1)

        return self.latent(x)

    def forward(self, x):

        return self.decoder(
            self.encode(x)
        )


mobile_classifier = nn.Sequential(

    nn.Linear(8, 32),
    nn.ReLU(),

    nn.Linear(32, 1)
)


# EXACT 30-feature order used by the mobile model.
MOBILE_FEATURE_COLUMNS = [

    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count"
]


class MobileDetector:

    def __init__(self):

        self.model_dir = "mobile_model"

        self.model_file = os.path.join(
            self.model_dir,
            "mobile_model.pth"
        )

        self.classifier_file = os.path.join(
            self.model_dir,
            "mobile_classifier.pth"
        )

        self.scaler_file = os.path.join(
            self.model_dir,
            "mobile_scaler.pkl"
        )

        self.threshold_file = os.path.join(
            self.model_dir,
            "mobile_threshold.pkl"
        )

        self.runtime_threshold_file = os.path.join(
            self.model_dir,
            "mobile_runtime_threshold.pkl"
        )

        self.model = MobileModel(30).to(DEVICE)
        self.classifier = mobile_classifier.to(DEVICE)

        self.flows = {}

        self.calibrating = True
        self.calibration_start = None
        self.calibration_errors = []
        self.threshold = None

        self.load()


    def load(self):

        print()
        print("[MOBILE] Loading model...")

        required_files = [
            self.model_file,
            self.classifier_file,
            self.scaler_file,
            self.threshold_file
        ]

        missing = [
            f for f in required_files
            if not os.path.exists(f)
        ]

        if missing:

            raise FileNotFoundError(
                "[MOBILE] Missing model files:\n"
                + "\n".join(missing)
                + "\n\nTrain the mobile model first."
            )

        self.model.load_state_dict(
            torch.load(
                self.model_file,
                map_location=DEVICE
            )
        )

        self.model.eval()

        self.classifier.load_state_dict(
            torch.load(
                self.classifier_file,
                map_location=DEVICE
            )
        )

        self.classifier.eval()

        with open(
            self.scaler_file,
            "rb"
        ) as f:

            self.scaler = pickle.load(f)

        with open(
            self.threshold_file,
            "rb"
        ) as f:

            self.training_threshold = float(
                pickle.load(f)
            )

        self.threshold = self.training_threshold

        print("[MOBILE] Model loaded.")
        print(
            "[MOBILE] Training threshold:",
            f"{self.training_threshold:.6f}"
        )


    # Mobile uses the same 30-flow-feature representation as
    # the separately trained mobile model.

    def flow_key(self, pkt):

        src = pkt[IP].src
        dst = pkt[IP].dst
        proto = pkt[IP].proto

        sport = 0
        dport = 0

        if pkt.haslayer(TCP):

            sport = int(pkt[TCP].sport)
            dport = int(pkt[TCP].dport)

        elif pkt.haslayer(UDP):

            sport = int(pkt[UDP].sport)
            dport = int(pkt[UDP].dport)

        a = (
            src,
            sport,
            dst,
            dport,
            proto
        )

        b = (
            dst,
            dport,
            src,
            sport,
            proto
        )

        return min(a, b)


    def create_flow(self, pkt):

        sport = 0
        dport = 0

        if pkt.haslayer(TCP):

            sport = int(pkt[TCP].sport)
            dport = int(pkt[TCP].dport)

        elif pkt.haslayer(UDP):

            sport = int(pkt[UDP].sport)
            dport = int(pkt[UDP].dport)

        timestamp = float(pkt.time)

        return {
            "start_time": timestamp,
            "last_seen": timestamp,

            "src_ip": pkt[IP].src,
            "dst_ip": pkt[IP].dst,

            "src_port": sport,
            "dst_port": dport,

            "fwd_times": [],
            "bwd_times": [],

            "fwd_lengths": [],
            "bwd_lengths": [],

            "fin": 0,
            "syn": 0,
            "rst": 0
        }


    def update_flow(self, flow, pkt):

        now = float(pkt.time)

        flow["last_seen"] = now

        length = len(pkt)

        is_forward = (
            pkt[IP].src == flow["src_ip"]
        )

        if is_forward:

            flow["fwd_times"].append(now)
            flow["fwd_lengths"].append(length)

        else:

            flow["bwd_times"].append(now)
            flow["bwd_lengths"].append(length)

        if pkt.haslayer(TCP):

            flags = int(pkt[TCP].flags)

            if flags & 0x01:
                flow["fin"] += 1

            if flags & 0x02:
                flow["syn"] += 1

            if flags & 0x04:
                flow["rst"] += 1


    def build_features(self, flow):

        fwd = flow["fwd_lengths"]
        bwd = flow["bwd_lengths"]

        all_lengths = fwd + bwd

        fwd_times = flow["fwd_times"]
        bwd_times = flow["bwd_times"]

        all_times = fwd_times + bwd_times

        duration = max(
            0.0,
            flow["last_seen"] - flow["start_time"]
        )

        total_fwd = len(fwd)
        total_bwd = len(bwd)

        total_fwd_bytes = sum(fwd)
        total_bwd_bytes = sum(bwd)

        total_packets = total_fwd + total_bwd
        total_bytes = total_fwd_bytes + total_bwd_bytes

        all_iat = inter_arrival_times(all_times)
        fwd_iat = inter_arrival_times(fwd_times)
        bwd_iat = inter_arrival_times(bwd_times)

        if duration > 0:

            flow_bytes_sec = total_bytes / duration
            flow_packets_sec = total_packets / duration
            fwd_packets_sec = total_fwd / duration
            bwd_packets_sec = total_bwd / duration

        else:

            flow_bytes_sec = 0.0
            flow_packets_sec = 0.0
            fwd_packets_sec = 0.0
            bwd_packets_sec = 0.0

        features = [

            flow["dst_port"],
            duration,

            total_fwd,
            total_bwd,

            total_fwd_bytes,
            total_bwd_bytes,

            safe_max(fwd),
            safe_mean(fwd),

            safe_max(bwd),
            safe_mean(bwd),

            flow_bytes_sec,
            flow_packets_sec,

            safe_mean(all_iat),
            safe_std(all_iat),
            safe_max(all_iat),
            safe_min(all_iat),

            safe_mean(fwd_iat),
            safe_std(fwd_iat),

            safe_mean(bwd_iat),
            safe_std(bwd_iat),

            fwd_packets_sec,
            bwd_packets_sec,

            safe_min(all_lengths),
            safe_max(all_lengths),
            safe_mean(all_lengths),
            safe_std(all_lengths),

            float(np.var(all_lengths))
            if all_lengths else 0.0,

            flow["fin"],
            flow["syn"],
            flow["rst"]
        ]

        return np.asarray(
            features,
            dtype=np.float32
        ).reshape(1, -1)


    def infer(self, flow):

        sample = self.build_features(flow)

        if sample.shape != (1, 30):

            raise RuntimeError(
                f"[MOBILE] Bad feature shape: {sample.shape}"
            )

        sample = self.scaler.transform(sample)

        sample = np.clip(
            sample,
            -20,
            20
        )

        x = torch.tensor(
            sample,
            dtype=torch.float32,
            device=DEVICE
        )

        with torch.no_grad():

            z = self.model.encode(x)

            reconstruction = self.model.decoder(z)

            error = (
                (reconstruction - x) ** 2
            ).mean().item()

            probability = torch.sigmoid(
                self.classifier(z)
            ).item()

        return error, probability


    def calibrate(self, flow):

        try:

            error, _ = self.infer(flow)

            self.calibration_errors.append(error)

        except Exception as e:

            print(
                "[MOBILE CALIBRATION ERROR]",
                repr(e)
            )


    def finish_calibration(self):

        count = len(
            self.calibration_errors
        )

        print()
        print("=" * 70)
        print("MOBILE CALIBRATION COMPLETE")
        print("=" * 70)

        print(
            "Samples:",
            count
        )

        if count < MIN_CALIBRATION_SAMPLES:

            print(
                "WARNING: insufficient mobile calibration samples."
            )

            print(
                "Keeping training threshold:",
                f"{self.training_threshold:.6f}"
            )

        else:

            errors = np.asarray(
                self.calibration_errors,
                dtype=np.float64
            )

            mean_error = np.mean(errors)
            median_error = np.median(errors)
            std_error = np.std(errors)

            percentile = np.percentile(
                errors,
                CALIBRATION_PERCENTILE
            )

            sigma = (
                mean_error
                +
                3.0 * std_error
            )

            self.threshold = max(
                percentile,
                sigma
            )

            if (
                not np.isfinite(self.threshold)
                or
                self.threshold <= 0
            ):

                self.threshold = (
                    self.training_threshold
                )

            print(
                f"Mean Error       : {mean_error:.6f}"
            )

            print(
                f"Median Error     : {median_error:.6f}"
            )

            print(
                f"Std Error        : {std_error:.6f}"
            )

            print(
                f"99th Percentile  : {percentile:.6f}"
            )

            print(
                f"Mean + 3*Std     : {sigma:.6f}"
            )

            print(
                f"MOBILE RUNTIME THRESHOLD: {self.threshold:.6f}"
            )

            try:

                with open(
                    self.runtime_threshold_file,
                    "wb"
                ) as f:

                    pickle.dump(
                        float(self.threshold),
                        f
                    )

            except Exception as e:

                print(
                    "Could not save mobile runtime threshold:",
                    repr(e)
                )

        self.calibrating = False

        print()
        print(
            "Mobile calibration traffic was NOT classified."
        )

        print(
            "Mobile threshold is now FROZEN."
        )

        print(
            "MOBILE LIVE DETECTION STARTING."
        )

        print("=" * 70)


    def detect(self, flow):

        try:

            error, probability = self.infer(flow)

        except Exception as e:

            print(
                "[MOBILE DETECTION ERROR]",
                repr(e)
            )

            return

        prediction, reason, alarm_level = classify_detection(
            error,
            self.threshold,
            probability
        )

        print()
        print(
            f"[MOBILE] {prediction}"
        )

        print(
            f"{flow['src_ip']} -> "
            f"{flow['dst_ip']}"
        )

        print(
            f"Error={error:.6f} | "
            f"Threshold={self.threshold:.6f} | "
            f"Probability={probability:.4f}"
        )

        print(
            f"Reason: {reason}."
        )

        print(
            f"Alarm Level: {alarm_level}"
        )

        if prediction == "SUSPICIOUS":

            print(
                "Status: classifier-only detection; full attack confirmation not reached."
            )

        elif prediction == "ATTACK":

            print(
                "Status: attack/anomaly detected."
            )

        trigger_alarm(alarm_level)

        db_queue.put(
            (
                dt.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                "MOBILE",

                flow["src_ip"],
                flow["dst_ip"],

                prediction,

                float(error),
                float(self.threshold),
                float(probability)
            )
        )


    def process_packet(self, pkt):

        if not pkt.haslayer(IP):
            return

        src = pkt[IP].src
        dst = pkt[IP].dst

        if (
            is_multicast_or_broadcast(src)
            or
            is_multicast_or_broadcast(dst)
        ):
            return

        key = self.flow_key(pkt)

        if key not in self.flows:

            self.flows[key] = (
                self.create_flow(pkt)
            )

        self.update_flow(
            self.flows[key],
            pkt
        )


    def expire_flows(self):

        now = time.time()

        expired = []

        for key, flow in list(
            self.flows.items()
        ):

            if (
                now - flow["last_seen"]
                >=
                FLOW_TIMEOUT
            ):

                if self.calibrating:

                    self.calibrate(flow)

                else:

                    self.detect(flow)

                expired.append(key)

        for key in expired:

            self.flows.pop(
                key,
                None
            )


    def run(self):

        print()
        print("=" * 70)
        print("MOBILE DATA / HOTSPOT MODE")
        print("=" * 70)

        print(
            "10-minutes calibration starting."
        )

        print(
            "Keep your mobile-data/hotspot traffic normal."
        )

        print(
            "Do NOT run ZAP/nmap/scans during calibration."
        )

        print()

        self.calibrating = True
        self.calibration_start = time.time()
        self.calibration_errors.clear()

        last_report = -1

        try:

            while self.calibrating:

                sniff(
                    iface=CAPTURE_INTERFACES,
                    prn=self.process_packet,
                    store=False,
                    filter="ip",
                    timeout=1
                )

                self.expire_flows()

                elapsed = (
                    time.time()
                    -
                    self.calibration_start
                )

                remaining = max(
                    0,
                    CALIBRATION_SECONDS
                    -
                    int(elapsed)
                )

                report = int(
                    elapsed // 10
                )

                if report != last_report:

                    last_report = report

                    print(
                        f"[MOBILE CALIBRATION] "
                        f"{remaining:03d}s remaining | "
                        f"flows={len(self.calibration_errors)}"
                    )

                if elapsed >= CALIBRATION_SECONDS:

                    self.finish_calibration()

            print()

            while True:

                sniff(
                    iface=CAPTURE_INTERFACES,
                    prn=self.process_packet,
                    store=False,
                    filter="ip",
                    timeout=1
                )

                self.expire_flows()

        except KeyboardInterrupt:

            print(
                "\nMobile IDS stopped."
            )



# ============================================================
# IoT MODEL
#
# This branch follows the IoT live/model implementation
# recovered from the earlier project code:
# 17 features -> CNN -> Transformer -> 8D latent -> decoder.
# ============================================================

class IoTModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv1d(
                1,
                32,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                32,
                64,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(16)
        )

        self.fc = nn.Linear(
            64 * 16,
            64
        )

        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            2
        )

        self.latent = nn.Linear(
            64,
            8
        )

        self.decoder = nn.Sequential(

            nn.Linear(
                8,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                17
            )
        )


    def encode(self, x):

        x = x.unsqueeze(1)

        x = self.cnn(x).flatten(1)

        x = self.fc(x)

        x = self.transformer(
            x.unsqueeze(1)
        ).squeeze(1)

        return self.latent(x)


    def forward(self, x):

        return self.decoder(
            self.encode(x)
        )


# ============================================================
# IoT CLASSIFIER
# ============================================================

iot_classifier = nn.Sequential(

    nn.Linear(
        8,
        32
    ),

    nn.ReLU(),

    nn.Linear(
        32,
        1
    )
)


# ============================================================
# IoT DETECTOR
#
# The recovered IoT live implementation used a lightweight
# packet-level 17-feature representation.
# ============================================================

class IoTDetector:

    def __init__(self):

        self.model_dir = "iot_model"

        self.model_file = os.path.join(
            self.model_dir,
            "model.pth"
        )

        self.classifier_file = os.path.join(
            self.model_dir,
            "classifier.pth"
        )

        self.scaler_file = os.path.join(
            self.model_dir,
            "scaler.pkl"
        )

        self.threshold_file = os.path.join(
            self.model_dir,
            "threshold.pkl"
        )

        self.runtime_threshold_file = os.path.join(
            self.model_dir,
            "runtime_threshold.pkl"
        )

        self.model = IoTModel().to(DEVICE)

        self.classifier = iot_classifier.to(DEVICE)

        self.calibrating = True

        self.calibration_start = None

        self.calibration_errors = []

        self.threshold = None

        self.load()


    def load(self):

        print()
        print("[IoT] Loading model...")

        self.model.load_state_dict(
            torch.load(
                self.model_file,
                map_location=DEVICE
            )
        )

        self.model.eval()

        self.classifier.load_state_dict(
            torch.load(
                self.classifier_file,
                map_location=DEVICE
            )
        )

        self.classifier.eval()

        with open(
            self.scaler_file,
            "rb"
        ) as f:

            self.scaler = pickle.load(f)

        try:

            with open(
                self.threshold_file,
                "rb"
            ) as f:

                self.training_threshold = float(
                    pickle.load(f)
                )

        except Exception:

            self.training_threshold = float("inf")

        self.threshold = (
            self.training_threshold
        )

        print(
            "[IoT] Model loaded."
        )

        print(
            "[IoT] Training threshold:",
            f"{self.training_threshold:.6f}"
        )


    def build_features(self, pkt):

        size = float(
            len(pkt)
        )

        time_ref = float(
            pkt.time % 1000
        )

        tcp = 1.0 if pkt.haslayer(TCP) else 0.0
        udp = 1.0 if pkt.haslayer(UDP) else 0.0
        icmp = 1.0 if pkt.haslayer(ICMP) else 0.0

        payload = float(
            len(pkt.payload)
        )

        ttl = float(
            pkt[IP].ttl
        )

        # The recovered IoT live pipeline used the first
        # seven packet-level values and zero-filled to 17.
        features = [
            size,
            time_ref,
            tcp,
            udp,
            icmp,
            payload,
            ttl
        ]

        while len(features) < 17:

            features.append(0.0)

        return np.asarray(
            features,
            dtype=np.float32
        ).reshape(
            1,
            -1
        )


    def infer(self, pkt):

        sample = self.build_features(pkt)

        if sample.shape != (1, 17):

            raise RuntimeError(
                f"[IoT] Bad feature shape: {sample.shape}"
            )

        sample = self.scaler.transform(
            sample
        )

        sample = np.clip(
            sample,
            -20,
            20
        )

        x = torch.tensor(
            sample,
            dtype=torch.float32,
            device=DEVICE
        )

        with torch.no_grad():

            z = self.model.encode(x)

            reconstruction = self.model(x)

            error = (
                (reconstruction - x) ** 2
            ).mean().item()

            probability = torch.sigmoid(
                self.classifier(z)
            ).item()

        return error, probability


    def calibrate(self, pkt):

        try:

            error, _ = self.infer(pkt)

            self.calibration_errors.append(
                error
            )

        except Exception as e:

            print(
                "[IoT CALIBRATION ERROR]",
                repr(e)
            )


    def finish_calibration(self):

        count = len(
            self.calibration_errors
        )

        print()
        print("=" * 70)
        print("IoT CALIBRATION COMPLETE")
        print("=" * 70)

        print(
            "Samples:",
            count
        )

        if count < MIN_CALIBRATION_SAMPLES:

            print(
                "WARNING: insufficient IoT calibration samples."
            )

            print(
                "Keeping training threshold:",
                f"{self.training_threshold:.6f}"
            )

        else:

            errors = np.asarray(
                self.calibration_errors,
                dtype=np.float64
            )

            mean_error = np.mean(errors)
            median_error = np.median(errors)
            std_error = np.std(errors)

            percentile = np.percentile(
                errors,
                CALIBRATION_PERCENTILE
            )

            sigma = (
                mean_error
                +
                3.0 * std_error
            )

            self.threshold = max(
                percentile,
                sigma
            )

            if (
                not np.isfinite(self.threshold)
                or
                self.threshold <= 0
            ):

                self.threshold = (
                    self.training_threshold
                )

            print(
                f"Mean Error       : {mean_error:.6f}"
            )

            print(
                f"Median Error     : {median_error:.6f}"
            )

            print(
                f"Std Error        : {std_error:.6f}"
            )

            print(
                f"99th Percentile  : {percentile:.6f}"
            )

            print(
                f"Mean + 3*Std     : {sigma:.6f}"
            )

            print(
                f"RUNTIME THRESHOLD: {self.threshold:.6f}"
            )

            try:

                with open(
                    self.runtime_threshold_file,
                    "wb"
                ) as f:

                    pickle.dump(
                        float(self.threshold),
                        f
                    )

            except Exception as e:

                print(
                    "Could not save IoT runtime threshold:",
                    repr(e)
                )

        self.calibrating = False

        print()
        print(
            "Calibration traffic was NOT classified."
        )

        print(
            "IoT threshold is now FROZEN."
        )

        print(
            "IoT LIVE DETECTION STARTING."
        )

        print("=" * 70)


    def detect(self, pkt):

        try:

            error, probability = self.infer(pkt)

        except Exception as e:

            print(
                "[IoT DETECTION ERROR]",
                repr(e)
            )

            return

        prediction, reason, alarm_level = classify_detection(
            error,
            self.threshold,
            probability
        )

        print()
        print(
            f"[IoT] {prediction}"
        )

        print(
            f"{pkt[IP].src} -> "
            f"{pkt[IP].dst}"
        )

        print(
            f"Error={error:.6f} | "
            f"Threshold={self.threshold:.6f} | "
            f"Probability={probability:.4f}"
        )

        print(
            f"Reason: {reason}."
        )

        print(
            f"Alarm Level: {alarm_level}"
        )

        if prediction == "SUSPICIOUS":
            print(
                "Status: classifier-only detection; full attack confirmation not reached."
            )
        elif prediction == "ATTACK":
            print(
                "Status: attack/anomaly detected."
            )

        trigger_alarm(alarm_level)

        db_queue.put(
            (
                dt.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                "IoT",

                pkt[IP].src,
                pkt[IP].dst,

                prediction,

                float(error),
                float(self.threshold),
                float(probability)
            )
        )


    def process_packet(self, pkt):

        if not pkt.haslayer(IP):

            return

        if (
            is_multicast_or_broadcast(
                pkt[IP].src
            )
            or
            is_multicast_or_broadcast(
                pkt[IP].dst
            )
        ):

            return

        if self.calibrating:

            self.calibrate(pkt)

        else:

            self.detect(pkt)


    def run(self):

        print()
        print("=" * 70)
        print("IoT MODE")
        print("=" * 70)

        print(
            "2-minute calibration starting."
        )

        print(
            "Keep IoT/network traffic normal."
        )

        print(
            "Do NOT run ZAP/nmap/scans during calibration."
        )

        print()

        self.calibrating = True
        self.calibration_start = time.time()

        self.calibration_errors.clear()

        last_report = -1

        try:

            while self.calibrating:

                sniff(
                    iface=CAPTURE_INTERFACES,
                    prn=self.process_packet,
                    store=False,
                    filter="ip",
                    timeout=1
                )

                elapsed = (
                    time.time()
                    -
                    self.calibration_start
                )

                remaining = max(
                    0,
                    CALIBRATION_SECONDS
                    -
                    int(elapsed)
                )

                report = int(
                    elapsed // 10
                )

                if report != last_report:

                    last_report = report

                    print(
                        f"[IoT CALIBRATION] "
                        f"{remaining:03d}s remaining | "
                        f"samples={len(self.calibration_errors)}"
                    )

                if elapsed >= CALIBRATION_SECONDS:

                    self.finish_calibration()

            print()

            print(
                "Listening for live IoT traffic..."
            )

            print(
                "Press CTRL+C to stop."
            )

            print()

            sniff(
                iface=CAPTURE_INTERFACES,
                prn=self.process_packet,
                store=False,
                filter="ip"
            )

        except KeyboardInterrupt:

            print(
                "\nIoT IDS stopped."
            )


# ============================================================
# MAIN
# ============================================================

def main():

    choice = choose_mode()

    if choice == "4":

        print(
            "Exiting."
        )

        return

    if choice == "1":

        detector = NetworkDetector()
        detector.run()

    elif choice == "2":

        detector = MobileDetector()
        detector.run()

    elif choice == "3":

        detector = IoTDetector()
        detector.run()


if __name__ == "__main__":

    main()