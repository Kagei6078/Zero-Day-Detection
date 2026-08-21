import os
import pickle
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "Network_dataset_1.csv"       # CHANGE THIS to your actual filename

OUTPUT_DIR = "iot_model"

BATCH_SIZE = 512
EPOCHS_AE = 15
EPOCHS_CLF = 15
LEARNING_RATE = 1e-3

LATENT_SIZE = 8

RANDOM_STATE = 42

# Exactly 17 IoT features
FEATURE_COLUMNS = [
    "ts",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "proto",
    "service",
    "duration",
    "src_bytes",
    "dst_bytes",
    "conn_state",
    "missed_bytes",
    "src_pkts",
    "src_ip_bytes",
    "dst_pkts",
    "dst_ip_bytes",
    "dns_qclass"
]

CATEGORICAL_COLUMNS = [
    "src_ip",
    "dst_ip",
    "proto",
    "service",
    "conn_state"
]

# Dataset label
POSSIBLE_LABEL_COLUMNS = [
    "type",
    "label",
    "attack",
    "class"
]


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("TON_IoT - 17 FEATURE HYBRID IDS TRAINING")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")
print(DATASET)

df = pd.read_csv(
    DATASET,
    low_memory=False
)

print("Dataset shape:", df.shape)


# ============================================================
# FIND LABEL COLUMN
# ============================================================

label_column = None

for col in POSSIBLE_LABEL_COLUMNS:
    if col in df.columns:
        label_column = col
        break

if label_column is None:
    raise RuntimeError(
        "Could not find label column. "
        "Expected one of: "
        + str(POSSIBLE_LABEL_COLUMNS)
    )

print("Label column:", label_column)


# ============================================================
# VERIFY 17 FEATURES
# ============================================================

missing_features = [
    col
    for col in FEATURE_COLUMNS
    if col not in df.columns
]

if missing_features:
    raise RuntimeError(
        "Missing required features:\n"
        + "\n".join(missing_features)
    )

print("\n17 FEATURES:")
for i, feature in enumerate(FEATURE_COLUMNS, 1):
    print(f"{i:2}. {feature}")

print("\nFeature count:", len(FEATURE_COLUMNS))

if len(FEATURE_COLUMNS) != 17:
    raise RuntimeError("Feature count is not 17!")


# ============================================================
# CLEAN LABELS
# ============================================================

df[label_column] = (
    df[label_column]
    .astype(str)
    .str.strip()
    .str.lower()
)

print("\nLabel distribution:")
print(df[label_column].value_counts())


# ============================================================
# CREATE BINARY LABEL
# ============================================================
#
# NORMAL = 0
# ATTACK = 1
#
# Anything other than "normal" is treated as attack.
# ============================================================

df["binary_label"] = (
    df[label_column] != "normal"
).astype(np.int64)

print("\nBinary distribution:")
print(df["binary_label"].value_counts())


# ============================================================
# COPY FEATURES
# ============================================================

data = df[FEATURE_COLUMNS].copy()


# ============================================================
# HANDLE CATEGORICAL FEATURES
# ============================================================

encoders = {}

print("\nEncoding categorical features...")

for column in CATEGORICAL_COLUMNS:

    encoder = LabelEncoder()

    data[column] = (
        data[column]
        .fillna("UNKNOWN")
        .astype(str)
    )

    data[column] = encoder.fit_transform(
        data[column]
    )

    encoders[column] = encoder

    print(
        f"{column}: "
        f"{len(encoder.classes_)} categories"
    )


# ============================================================
# CONVERT REMAINING FEATURES TO NUMERIC
# ============================================================

for column in FEATURE_COLUMNS:

    if column not in CATEGORICAL_COLUMNS:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

        data[column] = data[column].fillna(0)


# ============================================================
# NUMERICAL CLEANUP
# ============================================================

data = data.replace(
    [np.inf, -np.inf],
    0
)

data = data.fillna(0)


# ============================================================
# CONVERT TO FLOAT32
# ============================================================

X = data.astype(np.float32).values

y = df["binary_label"].values.astype(np.int64)

print("\nX shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTrain samples:", len(X_train))
print("Validation samples:", len(X_val))


# ============================================================
# SCALER
# ============================================================

print("\nFitting RobustScaler...")

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Same clipping philosophy as your existing pipeline
X_train = np.clip(
    X_train,
    -20,
    20
)

X_val = np.clip(
    X_val,
    -20,
    20
)

with open(
    os.path.join(
        OUTPUT_DIR,
        "iot_scaler.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        scaler,
        f
    )


# Save feature order
with open(
    os.path.join(
        OUTPUT_DIR,
        "iot_feature_columns.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        FEATURE_COLUMNS,
        f
    )


# Save encoders
with open(
    os.path.join(
        OUTPUT_DIR,
        "iot_encoders.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        encoders,
        f
    )


# ============================================================
# TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val,
    dtype=torch.float32
)


# ============================================================
# MODEL
# ============================================================

class HybridIoTModel(nn.Module):

    def __init__(
        self,
        feature_count=17,
        latent_size=8
    ):

        super().__init__()

        self.feature_count = feature_count

        # CNN
        self.cnn = nn.Sequential(

            nn.Conv1d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(32),

            nn.Conv1d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(64),

            nn.AdaptiveAvgPool1d(16)
        )


        # CNN → Transformer dimension
        self.fc = nn.Linear(
            64 * 16,
            64
        )


        # Transformer
        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                batch_first=True
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=2
            )
        )


        # Latent representation
        self.latent = nn.Linear(
            64,
            latent_size
        )


        # Decoder
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


    def encode(self, x):

        # [batch, features]
        x = x.unsqueeze(1)

        # CNN
        x = self.cnn(x)

        # Flatten
        x = x.flatten(1)

        # FC
        x = self.fc(x)

        # Transformer sequence
        x = x.unsqueeze(1)

        x = self.transformer(x)

        x = x.squeeze(1)

        # Latent
        z = self.latent(x)

        return z


    def forward(self, x):

        z = self.encode(x)

        reconstruction = self.decoder(z)

        return reconstruction


# ============================================================
# CREATE MODEL
# ============================================================

model = HybridIoTModel(
    feature_count=17,
    latent_size=LATENT_SIZE
).to(device)


# ============================================================
# AUTOENCODER TRAINING
# ============================================================
#
# IMPORTANT:
# Autoencoder sees NORMAL traffic only.
# ============================================================

normal_train = X_train[
    y_train == 0
]

normal_tensor = torch.tensor(
    normal_train,
    dtype=torch.float32
)

normal_loader = DataLoader(
    TensorDataset(normal_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("\nNormal training samples:",
      len(normal_train))


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.MSELoss()


print("\n" + "=" * 70)
print("TRAINING AUTOENCODER")
print("=" * 70)


for epoch in range(EPOCHS_AE):

    model.train()

    total_loss = 0.0

    for (batch,) in normal_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        reconstruction = model(batch)

        loss = criterion(
            reconstruction,
            batch
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * batch.size(0)
        )

    avg_loss = (
        total_loss
        / len(normal_train)
    )

    print(
        f"Epoch "
        f"{epoch + 1:02d}/"
        f"{EPOCHS_AE} "
        f"| Loss={avg_loss:.6f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "iot_model.pth"
)

torch.save(
    model.state_dict(),
    model_path
)

print("\nSaved:", model_path)


# ============================================================
# CALCULATE NORMAL RECONSTRUCTION ERRORS
# ============================================================

model.eval()

normal_val = X_val[
    y_val == 0
]

normal_val_tensor = torch.tensor(
    normal_val,
    dtype=torch.float32
)

errors = []

with torch.no_grad():

    for start in range(
        0,
        len(normal_val_tensor),
        BATCH_SIZE
    ):

        batch = (
            normal_val_tensor[
                start:start + BATCH_SIZE
            ]
            .to(device)
        )

        reconstruction = model(batch)

        batch_errors = (
            (reconstruction - batch) ** 2
        ).mean(dim=1)

        errors.extend(
            batch_errors.cpu().numpy()
        )


errors = np.array(errors)


# ============================================================
# THRESHOLD
# ============================================================

threshold = float(
    np.percentile(
        errors,
        99
    )
)

print("\n" + "=" * 70)
print("IO T ANOMALY THRESHOLD")
print("=" * 70)

print("Normal validation samples:",
      len(errors))

print("Mean error:",
      float(np.mean(errors)))

print("Std error:",
      float(np.std(errors)))

print("99th percentile:",
      threshold)


with open(
    os.path.join(
        OUTPUT_DIR,
        "iot_threshold.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        np.float32(threshold),
        f
    )


# ============================================================
# CLASSIFIER
# ============================================================

classifier = nn.Sequential(

    nn.Linear(
        LATENT_SIZE,
        32
    ),

    nn.ReLU(),

    nn.Linear(
        32,
        1
    )
).to(device)


# ============================================================
# CLASSIFIER DATA
# ============================================================

classifier_loader = DataLoader(

    TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32
        ),

        torch.tensor(
            y_train,
            dtype=torch.float32
        )
    ),

    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# CLASS WEIGHT
# ============================================================

positive_count = np.sum(
    y_train == 1
)

negative_count = np.sum(
    y_train == 0
)

if positive_count > 0:

    pos_weight = (
        negative_count
        / positive_count
    )

else:

    pos_weight = 1.0


pos_weight_tensor = torch.tensor(
    [pos_weight],
    dtype=torch.float32,
    device=device
)


classifier_optimizer = torch.optim.Adam(
    classifier.parameters(),
    lr=LEARNING_RATE
)

classifier_loss = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight_tensor
)


print("\n" + "=" * 70)
print("TRAINING ATTACK CLASSIFIER")
print("=" * 70)

for epoch in range(EPOCHS_CLF):

    classifier.train()

    total_loss = 0.0

    for batch_x, batch_y in classifier_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        classifier_optimizer.zero_grad()

        with torch.no_grad():

            z = model.encode(
                batch_x
            )

        logits = classifier(
            z
        ).squeeze(1)

        loss = classifier_loss(
            logits,
            batch_y
        )

        loss.backward()

        classifier_optimizer.step()

        total_loss += (
            loss.item()
            * batch_x.size(0)
        )

    avg_loss = (
        total_loss
        / len(X_train)
    )

    print(
        f"Epoch "
        f"{epoch + 1:02d}/"
        f"{EPOCHS_CLF} "
        f"| Loss={avg_loss:.6f}"
    )


# ============================================================
# SAVE CLASSIFIER
# ============================================================

classifier_path = os.path.join(
    OUTPUT_DIR,
    "iot_classifier.pth"
)

torch.save(
    classifier.state_dict(),
    classifier_path
)

print("\nSaved:", classifier_path)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

model.eval()
classifier.eval()

all_predictions = []
all_probabilities = []

with torch.no_grad():

    for start in range(
        0,
        len(X_val),
        BATCH_SIZE
    ):

        batch = torch.tensor(
            X_val[
                start:start + BATCH_SIZE
            ],
            dtype=torch.float32,
            device=device
        )

        z = model.encode(batch)

        logits = classifier(
            z
        ).squeeze(1)

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= 0.5
        ).long()

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


print(
    classification_report(
        y_val,
        all_predictions,
        target_names=[
            "NORMAL",
            "ATTACK"
        ],
        zero_division=0
    )
)


print("Confusion Matrix:")

print(
    confusion_matrix(
        y_val,
        all_predictions
    )
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "feature_count": 17,

    "feature_columns":
        FEATURE_COLUMNS,

    "categorical_columns":
        CATEGORICAL_COLUMNS,

    "latent_size":
        LATENT_SIZE,

    "attack_probability_threshold":
        0.50,

    "anomaly_threshold":
        threshold
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "iot_metadata.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        metadata,
        f
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("IOT MODEL TRAINING COMPLETE")
print("=" * 70)

print("\nFiles created:")

for filename in os.listdir(
    OUTPUT_DIR
):

    print(
        " -",
        os.path.join(
            OUTPUT_DIR,
            filename
        )
    )

print("\n17-feature IoT model is ready.")