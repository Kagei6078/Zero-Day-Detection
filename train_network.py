import os
import pickle
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

NORMAL_FILE = "real_network_normal.csv"
ATTACK_FILE = "real_network_attack.csv"

OUTPUT_DIR = "network_model"

FEATURE_COUNT = 30
LATENT_SIZE = 16

BATCH_SIZE = 1024

AE_EPOCHS = 80
CLASSIFIER_EPOCHS = 80

LEARNING_RATE = 1e-3

RANDOM_STATE = 42

THRESHOLD_PERCENTILE = 99.0

CLIP_VALUE = 20.0

# Early stopping
PATIENCE = 5


# ============================================================
# EXACT 30 FEATURES
# ============================================================

FEATURE_COLUMNS = [

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
# SEED
# ============================================================

random.seed(RANDOM_STATE)

np.random.seed(
    RANDOM_STATE
)

torch.manual_seed(
    RANDOM_STATE
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_STATE
    )


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# START
# ============================================================

print("=" * 75)
print("REAL NETWORK - 30 FEATURE HYBRID IDS")
print("=" * 75)

print()

print(
    "Device:",
    device
)

if device.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()

print(
    "Feature count:",
    len(FEATURE_COLUMNS)
)

print(
    "Latent size:",
    LATENT_SIZE
)

print()


# ============================================================
# VERIFY FEATURES
# ============================================================

if len(FEATURE_COLUMNS) != FEATURE_COUNT:

    raise RuntimeError(
        f"Expected {FEATURE_COUNT} features, "
        f"but found {len(FEATURE_COLUMNS)}"
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD NORMAL DATA
# ============================================================

print("=" * 75)
print("LOADING NORMAL NETWORK DATA")
print("=" * 75)

print()

normal = pd.read_csv(
    NORMAL_FILE,
    low_memory=False
)

print(
    "Normal rows:",
    len(normal)
)


# ============================================================
# LOAD ATTACK DATA
# ============================================================

print()

print("=" * 75)
print("LOADING ATTACK DATA")
print("=" * 75)

print()

attack = pd.read_csv(
    ATTACK_FILE,
    low_memory=False
)

print(
    "Attack rows:",
    len(attack)
)


# ============================================================
# VERIFY FEATURES
# ============================================================

missing_normal = [
    feature
    for feature in FEATURE_COLUMNS
    if feature not in normal.columns
]

missing_attack = [
    feature
    for feature in FEATURE_COLUMNS
    if feature not in attack.columns
]


if missing_normal:

    raise RuntimeError(
        "Missing features in NORMAL dataset:\n"
        +
        "\n".join(
            missing_normal
        )
    )


if missing_attack:

    raise RuntimeError(
        "Missing features in ATTACK dataset:\n"
        +
        "\n".join(
            missing_attack
        )
    )


# ============================================================
# KEEP ONLY 30 FEATURES
# ============================================================

normal = normal[
    FEATURE_COLUMNS
].copy()

attack = attack[
    FEATURE_COLUMNS
].copy()


# ============================================================
# NUMERIC CONVERSION
# ============================================================

print()
print(
    "Cleaning data..."
)

for feature in FEATURE_COLUMNS:

    normal[feature] = pd.to_numeric(
        normal[feature],
        errors="coerce"
    )

    attack[feature] = pd.to_numeric(
        attack[feature],
        errors="coerce"
    )


# ============================================================
# REMOVE INF
# ============================================================

normal = normal.replace(
    [np.inf, -np.inf],
    np.nan
)

attack = attack.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# DROP INVALID ROWS
# ============================================================

normal_before = len(normal)

attack_before = len(attack)


normal = normal.dropna()

attack = attack.dropna()


print(
    "Normal invalid rows removed:",
    normal_before - len(normal)
)

print(
    "Attack invalid rows removed:",
    attack_before - len(attack)
)

print()

print(
    "Clean normal rows:",
    len(normal)
)

print(
    "Clean attack rows:",
    len(attack)
)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X_normal = normal[
    FEATURE_COLUMNS
].values.astype(
    np.float32
)

X_attack = attack[
    FEATURE_COLUMNS
].values.astype(
    np.float32
)


# ============================================================
# SPLIT NORMAL DATA
#
# This is important:
#
# Normal data is split independently so the autoencoder
# NEVER trains on validation/test normal traffic.
# ============================================================

X_normal_train, X_normal_temp = train_test_split(

    X_normal,

    test_size=0.30,

    random_state=RANDOM_STATE
)


X_normal_val, X_normal_test = train_test_split(

    X_normal_temp,

    test_size=0.50,

    random_state=RANDOM_STATE
)


print()
print("=" * 75)
print("NORMAL DATA SPLIT")
print("=" * 75)

print(
    "Normal training:",
    len(X_normal_train)
)

print(
    "Normal validation:",
    len(X_normal_val)
)

print(
    "Normal testing:",
    len(X_normal_test)
)


# ============================================================
# SPLIT ATTACK DATA
# ============================================================

X_attack_train, X_attack_temp = train_test_split(

    X_attack,

    test_size=0.30,

    random_state=RANDOM_STATE
)


X_attack_val, X_attack_test = train_test_split(

    X_attack_temp,

    test_size=0.50,

    random_state=RANDOM_STATE
)


print()
print("=" * 75)
print("ATTACK DATA SPLIT")
print("=" * 75)

print(
    "Attack training:",
    len(X_attack_train)
)

print(
    "Attack validation:",
    len(X_attack_val)
)

print(
    "Attack testing:",
    len(X_attack_test)
)


# ============================================================
# SCALER
#
# IMPORTANT:
# Fit ONLY on NORMAL TRAINING traffic.
#
# This makes the scaling representative of normal network
# behavior and avoids allowing attack traffic to define
# the normal baseline.
# ============================================================

scaler = RobustScaler()

scaler.fit(
    X_normal_train
)


# ============================================================
# TRANSFORM DATA
# ============================================================

X_normal_train_scaled = scaler.transform(
    X_normal_train
)

X_normal_val_scaled = scaler.transform(
    X_normal_val
)

X_normal_test_scaled = scaler.transform(
    X_normal_test
)

X_attack_train_scaled = scaler.transform(
    X_attack_train
)

X_attack_val_scaled = scaler.transform(
    X_attack_val
)

X_attack_test_scaled = scaler.transform(
    X_attack_test
)


# ============================================================
# CLIP EXTREME VALUES
# ============================================================

X_normal_train_scaled = np.clip(
    X_normal_train_scaled,
    -CLIP_VALUE,
    CLIP_VALUE
)

X_normal_val_scaled = np.clip(
    X_normal_val_scaled,
    -CLIP_VALUE,
    CLIP_VALUE
)

X_normal_test_scaled = np.clip(
    X_normal_test_scaled,
    -CLIP_VALUE,
    CLIP_VALUE
)

X_attack_train_scaled = np.clip(
    X_attack_train_scaled,
    -CLIP_VALUE,
    CLIP_VALUE
)

X_attack_val_scaled = np.clip(
    X_attack_val_scaled,
    -CLIP_VALUE,
    CLIP_VALUE
)

X_attack_test_scaled = np.clip(
    X_attack_test_scaled,
    -CLIP_VALUE,
    CLIP_VALUE
)


# ============================================================
# SAVE SCALER
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "network_scaler.pkl"
    ),
    "wb"
) as file:

    pickle.dump(
        scaler,
        file
    )


# ============================================================
# SAVE FEATURE ORDER
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "network_feature_columns.pkl"
    ),
    "wb"
) as file:

    pickle.dump(
        FEATURE_COLUMNS,
        file
    )


# ============================================================
# HYBRID CNN + TRANSFORMER + AUTOENCODER
# ============================================================

class HybridNetworkModel(nn.Module):

    def __init__(
        self,
        feature_count=30,
        latent_size=16
    ):

        super().__init__()


        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        self.cnn = nn.Sequential(

            nn.Conv1d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                32
            ),

            nn.Conv1d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                64
            ),

            nn.AdaptiveAvgPool1d(
                16
            )
        )


        # ----------------------------------------------------
        # CNN → FEATURE VECTOR
        # ----------------------------------------------------

        self.fc = nn.Linear(
            64 * 16,
            64
        )


        # ----------------------------------------------------
        # TRANSFORMER
        # ----------------------------------------------------

        transformer_layer = (
            nn.TransformerEncoderLayer(

                d_model=64,

                nhead=4,

                batch_first=True
            )
        )


        self.transformer = (
            nn.TransformerEncoder(

                transformer_layer,

                num_layers=2
            )
        )


        # ----------------------------------------------------
        # LATENT ENCODER
        # ----------------------------------------------------

        self.latent = nn.Linear(
            64,
            latent_size
        )


        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        self.decoder = nn.Sequential(

            nn.Linear(
                latent_size,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                feature_count
            )
        )


    # ========================================================
    # ENCODER
    # ========================================================

    def encode(
        self,
        x
    ):

        # [batch, 30]
        x = x.unsqueeze(1)

        # CNN
        x = self.cnn(
            x
        )

        # Flatten
        x = x.flatten(
            1
        )

        # Fully connected
        x = self.fc(
            x
        )

        # Transformer sequence
        x = x.unsqueeze(
            1
        )

        x = self.transformer(
            x
        )

        x = x.squeeze(
            1
        )

        # Latent representation
        z = self.latent(
            x
        )

        return z


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x
    ):

        z = self.encode(
            x
        )

        reconstruction = self.decoder(
            z
        )

        return reconstruction


# ============================================================
# CREATE MODEL
# ============================================================

model = HybridNetworkModel(
    feature_count=FEATURE_COUNT,
    latent_size=LATENT_SIZE
).to(
    device
)


# ============================================================
# AUTOENCODER DATASET
#
# ONLY NORMAL TRAINING DATA
# ============================================================

normal_train_tensor = torch.tensor(
    X_normal_train_scaled,
    dtype=torch.float32
)


normal_train_loader = DataLoader(

    TensorDataset(
        normal_train_tensor
    ),

    batch_size=BATCH_SIZE,

    shuffle=True,

    pin_memory=(
        device.type == "cuda"
    )
)


# ============================================================
# AUTOENCODER TRAINING
# ============================================================

print()
print("=" * 75)
print("TRAINING REAL-NETWORK AUTOENCODER")
print("=" * 75)

print()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

criterion = nn.MSELoss()


best_normal_val_loss = float(
    "inf"
)

best_model_state = None

patience_counter = 0


# Validation tensor
normal_val_tensor = torch.tensor(
    X_normal_val_scaled,
    dtype=torch.float32
)


for epoch in range(
    AE_EPOCHS
):

    model.train()

    total_loss = 0.0

    total_samples = 0


    for (batch,) in normal_train_loader:

        batch = batch.to(
            device,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        reconstruction = model(
            batch
        )


        loss = criterion(
            reconstruction,
            batch
        )


        loss.backward()

        optimizer.step()


        total_loss += (
            loss.item()
            *
            batch.size(0)
        )

        total_samples += batch.size(0)


    train_loss = (
        total_loss /
        total_samples
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_total = 0.0

    val_samples = 0


    with torch.no_grad():

        for start in range(
            0,
            len(normal_val_tensor),
            BATCH_SIZE
        ):

            batch = (
                normal_val_tensor[
                    start:
                    start + BATCH_SIZE
                ]
                .to(device)
            )


            reconstruction = model(
                batch
            )


            loss = criterion(
                reconstruction,
                batch
            )


            val_loss_total += (
                loss.item()
                *
                batch.size(0)
            )

            val_samples += batch.size(0)


    val_loss = (
        val_loss_total /
        val_samples
    )


    print(
        f"Epoch {epoch + 1:02d}/{AE_EPOCHS} "
        f"| Train Loss = {train_loss:.6f} "
        f"| Val Loss = {val_loss:.6f}"
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_normal_val_loss:

        best_normal_val_loss = val_loss

        best_model_state = {
            key: value.detach().cpu().clone()
            for key, value
            in model.state_dict().items()
        }

        patience_counter = 0

    else:

        patience_counter += 1


    if patience_counter >= PATIENCE:

        print(
            "Early stopping autoencoder."
        )

        break


# ============================================================
# RESTORE BEST AUTOENCODER
# ============================================================

if best_model_state is not None:

    model.load_state_dict(
        best_model_state
    )


# ============================================================
# SAVE NETWORK MODEL
# ============================================================

MODEL_FILE = os.path.join(
    OUTPUT_DIR,
    "network_model.pth"
)


torch.save(
    model.state_dict(),
    MODEL_FILE
)


print()
print(
    "Saved:",
    MODEL_FILE
)


# ============================================================
# CALCULATE NORMAL RECONSTRUCTION ERRORS
# ============================================================

print()
print("=" * 75)
print("CALCULATING NETWORK-SPECIFIC ANOMALY THRESHOLD")
print("=" * 75)

model.eval()

normal_errors = []


with torch.no_grad():

    for start in range(
        0,
        len(normal_val_tensor),
        BATCH_SIZE
    ):

        batch = (
            normal_val_tensor[
                start:
                start + BATCH_SIZE
            ]
            .to(device)
        )


        reconstruction = model(
            batch
        )


        errors = (
            (
                reconstruction -
                batch
            )
            ** 2
        ).mean(
            dim=1
        )


        normal_errors.extend(
            errors
            .cpu()
            .numpy()
        )


normal_errors = np.asarray(
    normal_errors
)


# ============================================================
# NETWORK THRESHOLD
# ============================================================

network_threshold = float(
    np.percentile(
        normal_errors,
        THRESHOLD_PERCENTILE
    )
)


print()

print(
    "Normal validation flows:",
    len(normal_errors)
)

print(
    "Mean reconstruction error:",
    np.mean(normal_errors)
)

print(
    "Median reconstruction error:",
    np.median(normal_errors)
)

print(
    "Std reconstruction error:",
    np.std(normal_errors)
)

print(
    "99th percentile threshold:",
    network_threshold
)


# ============================================================
# SAVE THRESHOLD
# ============================================================

THRESHOLD_FILE = os.path.join(
    OUTPUT_DIR,
    "network_threshold.pkl"
)


with open(
    THRESHOLD_FILE,
    "wb"
) as file:

    pickle.dump(
        np.float32(
            network_threshold
        ),
        file
    )


print()
print(
    "Saved:",
    THRESHOLD_FILE
)


# ============================================================
# LATENT CLASSIFIER
# ============================================================

classifier = nn.Sequential(

    nn.Linear(
        LATENT_SIZE,
        64
    ),

    nn.ReLU(),

    nn.Dropout(
        0.20
    ),

    nn.Linear(
        64,
        32
    ),

    nn.ReLU(),

    nn.Linear(
        32,
        1
    )
).to(
    device
)


# ============================================================
# CLASSIFIER DATA
#
# NORMAL + ATTACK
# ============================================================

X_classifier_train = np.vstack(
    [
        X_normal_train_scaled,
        X_attack_train_scaled
    ]
)


y_classifier_train = np.concatenate(
    [
        np.zeros(
            len(
                X_normal_train_scaled
            ),
            dtype=np.float32
        ),

        np.ones(
            len(
                X_attack_train_scaled
            ),
            dtype=np.float32
        )
    ]
)


# ------------------------------------------------------------
# Shuffle
# ------------------------------------------------------------

rng = np.random.default_rng(
    RANDOM_STATE
)

indices = rng.permutation(
    len(
        X_classifier_train
    )
)


X_classifier_train = (
    X_classifier_train[
        indices
    ]
)

y_classifier_train = (
    y_classifier_train[
        indices
    ]
)


# ============================================================
# CLASSIFIER VALIDATION DATA
# ============================================================

X_classifier_val = np.vstack(
    [
        X_normal_val_scaled,
        X_attack_val_scaled
    ]
)


y_classifier_val = np.concatenate(
    [
        np.zeros(
            len(
                X_normal_val_scaled
            ),
            dtype=np.float32
        ),

        np.ones(
            len(
                X_attack_val_scaled
            ),
            dtype=np.float32
        )
    ]
)


# ============================================================
# CLASSIFIER TEST DATA
# ============================================================

X_classifier_test = np.vstack(
    [
        X_normal_test_scaled,
        X_attack_test_scaled
    ]
)


y_classifier_test = np.concatenate(
    [
        np.zeros(
            len(
                X_normal_test_scaled
            ),
            dtype=np.float32
        ),

        np.ones(
            len(
                X_attack_test_scaled
            ),
            dtype=np.float32
        )
    ]
)


# ============================================================
# TORCH DATA
# ============================================================

classifier_train_tensor = torch.tensor(
    X_classifier_train,
    dtype=torch.float32
)

classifier_train_labels = torch.tensor(
    y_classifier_train,
    dtype=torch.float32
)


classifier_val_tensor = torch.tensor(
    X_classifier_val,
    dtype=torch.float32
)

classifier_val_labels = torch.tensor(
    y_classifier_val,
    dtype=torch.float32
)


classifier_test_tensor = torch.tensor(
    X_classifier_test,
    dtype=torch.float32
)

classifier_test_labels = torch.tensor(
    y_classifier_test,
    dtype=torch.float32
)


classifier_loader = DataLoader(

    TensorDataset(
        classifier_train_tensor,
        classifier_train_labels
    ),

    batch_size=BATCH_SIZE,

    shuffle=True,

    pin_memory=(
        device.type == "cuda"
    )
)


# ============================================================
# CLASS WEIGHT
# ============================================================

positive_count = np.sum(
    y_classifier_train == 1
)

negative_count = np.sum(
    y_classifier_train == 0
)


if positive_count > 0:

    pos_weight = (
        negative_count /
        positive_count
    )

else:

    pos_weight = 1.0


print()
print(
    "=" * 75
)

print(
    "TRAINING LATENT ATTACK CLASSIFIER"
)

print(
    "=" * 75
)

print()

print(
    "Normal classifier training:",
    negative_count
)

print(
    "Attack classifier training:",
    positive_count
)

print(
    "Positive class weight:",
    pos_weight
)


# ============================================================
# CLASSIFIER LOSS
# ============================================================

pos_weight_tensor = torch.tensor(
    [pos_weight],
    dtype=torch.float32,
    device=device
)


classifier_loss = (
    nn.BCEWithLogitsLoss(
        pos_weight=pos_weight_tensor
    )
)


classifier_optimizer = torch.optim.AdamW(
    classifier.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# TRAIN CLASSIFIER
#
# IMPORTANT:
# Encoder weights are FROZEN here.
# ============================================================

best_classifier_f1 = -1.0

best_classifier_state = None

classifier_patience = 0


for epoch in range(
    CLASSIFIER_EPOCHS
):

    classifier.train()

    total_loss = 0.0

    total_samples = 0


    for batch_x, batch_y in classifier_loader:

        batch_x = batch_x.to(
            device,
            non_blocking=True
        )

        batch_y = batch_y.to(
            device,
            non_blocking=True
        )


        classifier_optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # ENCODER
        # ----------------------------------------------------

        with torch.no_grad():

            latent = model.encode(
                batch_x
            )


        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        logits = classifier(
            latent
        ).squeeze(
            1
        )


        loss = classifier_loss(
            logits,
            batch_y
        )


        loss.backward()

        classifier_optimizer.step()


        total_loss += (
            loss.item()
            *
            batch_x.size(0)
        )

        total_samples += (
            batch_x.size(0)
        )


    train_loss = (
        total_loss /
        total_samples
    )


    # ========================================================
    # VALIDATE CLASSIFIER
    # ========================================================

    classifier.eval()

    val_predictions = []

    val_labels_list = []


    with torch.no_grad():

        for start in range(
            0,
            len(classifier_val_tensor),
            BATCH_SIZE
        ):

            batch_x = (
                classifier_val_tensor[
                    start:
                    start + BATCH_SIZE
                ]
                .to(device)
            )

            batch_y = (
                classifier_val_labels[
                    start:
                    start + BATCH_SIZE
                ]
                .to(device)
            )


            latent = model.encode(
                batch_x
            )


            logits = classifier(
                latent
            ).squeeze(
                1
            )


            probabilities = torch.sigmoid(
                logits
            )


            predictions = (
                probabilities >= 0.5
            ).long()


            val_predictions.extend(
                predictions
                .cpu()
                .numpy()
            )

            val_labels_list.extend(
                batch_y
                .cpu()
                .numpy()
                .astype(int)
            )


    val_predictions = np.asarray(
        val_predictions
    )

    val_labels_array = np.asarray(
        val_labels_list
    )


    val_f1 = f1_score(
        val_labels_array,
        val_predictions,
        zero_division=0
    )


    val_recall = recall_score(
        val_labels_array,
        val_predictions,
        zero_division=0
    )


    print(
        f"Epoch {epoch + 1:02d}/"
        f"{CLASSIFIER_EPOCHS}"
        f" | Loss = {train_loss:.6f}"
        f" | Val F1 = {val_f1:.4f}"
        f" | Val Recall = {val_recall:.4f}"
    )


    # --------------------------------------------------------
    # BEST CLASSIFIER
    # --------------------------------------------------------

    if val_f1 > best_classifier_f1:

        best_classifier_f1 = val_f1

        best_classifier_state = {
            key: value.detach().cpu().clone()
            for key, value
            in classifier.state_dict().items()
        }

        classifier_patience = 0

    else:

        classifier_patience += 1


    if classifier_patience >= PATIENCE:

        print(
            "Early stopping classifier."
        )

        break


# ============================================================
# RESTORE BEST CLASSIFIER
# ============================================================

if best_classifier_state is not None:

    classifier.load_state_dict(
        best_classifier_state
    )


# ============================================================
# SAVE CLASSIFIER
# ============================================================

CLASSIFIER_FILE = os.path.join(
    OUTPUT_DIR,
    "network_classifier.pth"
)


torch.save(
    classifier.state_dict(),
    CLASSIFIER_FILE
)


print()
print(
    "Saved:",
    CLASSIFIER_FILE
)


# ============================================================
# FINAL TEST
# ============================================================

model.eval()

classifier.eval()

test_predictions = []

test_probabilities = []


with torch.no_grad():

    for start in range(
        0,
        len(classifier_test_tensor),
        BATCH_SIZE
    ):

        batch_x = (
            classifier_test_tensor[
                start:
                start + BATCH_SIZE
            ]
            .to(device)
        )


        latent = model.encode(
            batch_x
        )


        logits = classifier(
            latent
        ).squeeze(
            1
        )


        probabilities = torch.sigmoid(
            logits
        )


        predictions = (
            probabilities >= 0.5
        ).long()


        test_predictions.extend(
            predictions
            .cpu()
            .numpy()
        )

        test_probabilities.extend(
            probabilities
            .cpu()
            .numpy()
        )


test_predictions = np.asarray(
    test_predictions
)

test_probabilities = np.asarray(
    test_probabilities
)


# ============================================================
# METRICS
# ============================================================

test_accuracy = accuracy_score(
    y_classifier_test,
    test_predictions
)

test_precision = precision_score(
    y_classifier_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_classifier_test,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_classifier_test,
    test_predictions,
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 75)
print("FINAL REAL-NETWORK CLASSIFIER TEST")
print("=" * 75)

print()

print(
    classification_report(
        y_classifier_test,
        test_predictions,
        target_names=[
            "NORMAL",
            "ATTACK"
        ],
        digits=4,
        zero_division=0
    )
)

print(
    "Accuracy :",
    f"{test_accuracy:.4f}"
)

print(
    "Precision:",
    f"{test_precision:.4f}"
)

print(
    "Recall   :",
    f"{test_recall:.4f}"
)

print(
    "F1       :",
    f"{test_f1:.4f}"
)

print()

print(
    "Confusion Matrix:"
)

print(
    confusion_matrix(
        y_classifier_test,
        test_predictions
    )
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "model_type":
        "Real Network CNN Transformer Autoencoder",

    "dataset":
        "real_network_normal + real_network_attack",

    "feature_count":
        FEATURE_COUNT,

    "feature_columns":
        FEATURE_COLUMNS,

    "latent_size":
        LATENT_SIZE,

    "threshold_percentile":
        THRESHOLD_PERCENTILE,

    "network_threshold":
        network_threshold,

    "classifier_probability_threshold":
        0.50,

    "clip_value":
        CLIP_VALUE,

    "normal_training_samples":
        len(X_normal_train),

    "normal_validation_samples":
        len(X_normal_val),

    "normal_test_samples":
        len(X_normal_test),

    "attack_training_samples":
        len(X_attack_train),

    "attack_validation_samples":
        len(X_attack_val),

    "attack_test_samples":
        len(X_attack_test),

    "classifier_validation_f1":
        float(best_classifier_f1),

    "classifier_test_accuracy":
        float(test_accuracy),

    "classifier_test_precision":
        float(test_precision),

    "classifier_test_recall":
        float(test_recall),

    "classifier_test_f1":
        float(test_f1)
}


METADATA_FILE = os.path.join(
    OUTPUT_DIR,
    "network_metadata.pkl"
)


with open(
    METADATA_FILE,
    "wb"
) as file:

    pickle.dump(
        metadata,
        file
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 75)
print("REAL NETWORK MODEL TRAINING COMPLETE")
print("=" * 75)

print()

print(
    "Generated files:"
)

print()

print(
    "  network_model\\network_model.pth"
)

print(
    "  network_model\\network_classifier.pth"
)

print(
    "  network_model\\network_scaler.pkl"
)

print(
    "  network_model\\network_feature_columns.pkl"
)

print(
    "  network_model\\network_threshold.pkl"
)

print(
    "  network_model\\network_metadata.pkl"
)

print()

print(
    "Network threshold:",
    network_threshold
)

print(
    "Classifier F1:",
    test_f1
)

print()

print(
    "CIC-IDS2017 is NOT used."
)

print(
    "The network model is trained on YOUR captured traffic."
)

print("=" * 75)