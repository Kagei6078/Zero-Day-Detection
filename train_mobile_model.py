import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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

from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

NORMAL_FILE = "real_network_mobile_normal.csv"
ATTACK_FILE = "real_network_mobile_attack.csv"

OUTPUT_DIR = "mobile_model_files"

EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 0.001

LATENT_SIZE = 8

# Probability used by live detection
ATTACK_PROBABILITY_THRESHOLD = 0.70

# Reconstruction threshold multiplier
THRESHOLD_K = 3.0

RANDOM_STATE = 42


# ============================================================
# 30 NETWORK FEATURES
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


FEATURE_COUNT = len(FEATURE_COLUMNS)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    DEVICE = torch.device("cpu")

    print("Using CPU")


print()
print("=" * 70)
print("MOBILE NETWORK HYBRID CNN + TRANSFORMER")
print("=" * 70)

print(
    "Feature count:",
    FEATURE_COUNT
)

print(
    "Device:",
    DEVICE
)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("Loading NORMAL dataset...")

normal_df = pd.read_csv(
    NORMAL_FILE,
    low_memory=False
)


print(
    "Normal samples:",
    len(normal_df)
)


print()
print("Loading ATTACK dataset...")

attack_df = pd.read_csv(
    ATTACK_FILE,
    low_memory=False
)


print(
    "Attack samples:",
    len(attack_df)
)


# ============================================================
# VERIFY FEATURES
# ============================================================

print()
print("Checking feature columns...")


missing_normal = [

    col
    for col in FEATURE_COLUMNS
    if col not in normal_df.columns

]


missing_attack = [

    col
    for col in FEATURE_COLUMNS
    if col not in attack_df.columns

]


if missing_normal:

    print()
    print("Missing from NORMAL:")

    for col in missing_normal:
        print(col)

    raise ValueError(
        "Normal dataset does not contain all 30 features."
    )


if missing_attack:

    print()
    print("Missing from ATTACK:")

    for col in missing_attack:
        print(col)

    raise ValueError(
        "Attack dataset does not contain all 30 features."
    )


print(
    "All 30 features found in both datasets."
)


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

print()
print("=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)


normal_before = len(normal_df)

attack_before = len(attack_df)


normal_df = normal_df.drop_duplicates(
    subset=FEATURE_COLUMNS
).reset_index(drop=True)


attack_df = attack_df.drop_duplicates(
    subset=FEATURE_COLUMNS
).reset_index(drop=True)


print(
    "Normal duplicates removed:",
    normal_before - len(normal_df)
)

print(
    "Attack duplicates removed:",
    attack_before - len(attack_df)
)


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    X = df[
        FEATURE_COLUMNS
    ].copy()


    # Convert to numeric

    for col in FEATURE_COLUMNS:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )


    # Replace infinity

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )


    # Replace missing

    X = X.fillna(0)


    return X.values.astype(
        np.float32
    )


normal_X = prepare_features(
    normal_df
)


attack_X = prepare_features(
    attack_df
)


print()
print(
    "Normal shape:",
    normal_X.shape
)

print(
    "Attack shape:",
    attack_X.shape
)


# ============================================================
# LABELS
# ============================================================

normal_y = np.zeros(
    len(normal_X),
    dtype=np.int64
)


attack_y = np.ones(
    len(attack_X),
    dtype=np.int64
)


# ============================================================
# COMBINE DATA
# ============================================================

X = np.concatenate(
    [
        normal_X,
        attack_X
    ],
    axis=0
)


y = np.concatenate(
    [
        normal_y,
        attack_y
    ],
    axis=0
)


print()
print("=" * 70)
print("DATASET")
print("=" * 70)

print(
    "Total:",
    len(X)
)

print(
    "NORMAL:",
    np.sum(y == 0)
)

print(
    "ATTACK:",
    np.sum(y == 1)
)


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


print()
print(
    "Training samples:",
    len(X_train)
)

print(
    "Validation samples:",
    len(X_val)
)


# ============================================================
# SCALER
# ============================================================

print()
print("Training RobustScaler...")


scaler = RobustScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_val_scaled = scaler.transform(
    X_val
)


# Limit extreme values

X_train_scaled = np.clip(
    X_train_scaled,
    -20,
    20
)


X_val_scaled = np.clip(
    X_val_scaled,
    -20,
    20
)


# ============================================================
# SAVE SCALER
# ============================================================

scaler_path = os.path.join(
    OUTPUT_DIR,
    "mobile_scaler.pkl"
)


with open(
    scaler_path,
    "wb"
) as f:

    pickle.dump(
        scaler,
        f
    )


print(
    "Saved:",
    scaler_path
)


# ============================================================
# HYBRID CNN + TRANSFORMER
# ============================================================

class HybridModel(nn.Module):

    def __init__(self, feature_count):

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

            nn.MaxPool1d(
                2
            ),


            nn.Conv1d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(
                16
            )

        )


        # ----------------------------------------------------
        # FULLY CONNECTED
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
        # LATENT SPACE
        # ----------------------------------------------------

        self.latent = nn.Linear(
            64,
            LATENT_SIZE
        )


        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        self.decoder = nn.Sequential(

            nn.Linear(
                LATENT_SIZE,
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

    def encode(self, x):

        # [batch, 30]
        x = x.unsqueeze(1)

        # [batch, 1, 30]
        x = self.cnn(x)

        # [batch, 64, 16]
        x = x.flatten(1)

        # [batch, 1024]
        x = self.fc(x)

        # [batch, 64]

        x = x.unsqueeze(1)

        # [batch, 1, 64]

        x = self.transformer(x)

        x = x.squeeze(1)

        # [batch, 64]

        z = self.latent(x)

        # [batch, 8]

        return z


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        z = self.encode(x)

        reconstruction = self.decoder(z)

        return reconstruction


# ============================================================
# CREATE MODEL
# ============================================================

model = HybridModel(
    FEATURE_COUNT
).to(DEVICE)


print()
print(
    "Hybrid model created."
)


# ============================================================
# AUTOENCODER DATA
# ============================================================

# IMPORTANT:
# Autoencoder sees NORMAL traffic only.

normal_training_mask = (
    y_train == 0
)


normal_train = X_train_scaled[
    normal_training_mask
]


print()
print(
    "Normal samples for autoencoder:",
    len(normal_train)
)


normal_dataset = TensorDataset(

    torch.tensor(
        normal_train,
        dtype=torch.float32
    )

)


normal_loader = DataLoader(

    normal_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True
)


# ============================================================
# AUTOENCODER TRAINING
# ============================================================

print()
print("=" * 70)
print("TRAINING AUTOENCODER")
print("=" * 70)


optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE

)


criterion = nn.MSELoss()


model.train()


for epoch in range(EPOCHS):

    total_loss = 0.0


    for batch in normal_loader:

        x = batch[0].to(
            DEVICE
        )


        optimizer.zero_grad()


        reconstruction = model(
            x
        )


        loss = criterion(
            reconstruction,
            x
        )


        loss.backward()


        optimizer.step()


        total_loss += (

            loss.item()
            *
            len(x)

        )


    average_loss = (

        total_loss
        /
        len(normal_dataset)

    )


    print(

        f"Epoch "
        f"{epoch + 1:02d}/{EPOCHS} "
        f"| Loss: "
        f"{average_loss:.6f}"

    )


# ============================================================
# RECONSTRUCTION ERRORS
# ============================================================

print()
print(
    "Calculating NORMAL reconstruction errors..."
)


model.eval()


normal_tensor = torch.tensor(

    normal_train,

    dtype=torch.float32

).to(DEVICE)


normal_errors = []


with torch.no_grad():

    for start in range(

        0,

        len(normal_tensor),

        BATCH_SIZE

    ):

        batch = normal_tensor[
            start:
            start + BATCH_SIZE
        ]


        reconstruction = model(
            batch
        )


        error = (

            (reconstruction - batch)
            ** 2

        ).mean(
            dim=1
        )


        normal_errors.extend(

            error
            .cpu()
            .numpy()
            .tolist()

        )


normal_errors = np.array(
    normal_errors
)


# ============================================================
# THRESHOLD
# ============================================================

mean_error = normal_errors.mean()

std_error = normal_errors.std()


threshold = (

    mean_error
    +
    THRESHOLD_K * std_error

)


print()
print("=" * 70)
print("RECONSTRUCTION THRESHOLD")
print("=" * 70)

print(
    f"Mean      : {mean_error:.6f}"
)

print(
    f"Std       : {std_error:.6f}"
)

print(
    f"K         : {THRESHOLD_K}"
)

print(
    f"Threshold : {threshold:.6f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(

    OUTPUT_DIR,

    "mobile_model.pth"

)


torch.save(

    model.state_dict(),

    model_path

)


print()
print(
    "Saved:",
    model_path
)


# ============================================================
# SAVE THRESHOLD
# ============================================================

threshold_path = os.path.join(

    OUTPUT_DIR,

    "mobile_threshold.pkl"

)


with open(

    threshold_path,

    "wb"

) as f:

    pickle.dump(

        float(threshold),

        f

    )


print(
    "Saved:",
    threshold_path
)


# ============================================================
# LATENT FEATURES
# ============================================================

print()
print(
    "Extracting latent features..."
)


X_train_tensor = torch.tensor(

    X_train_scaled,

    dtype=torch.float32

).to(DEVICE)


X_val_tensor = torch.tensor(

    X_val_scaled,

    dtype=torch.float32

).to(DEVICE)


with torch.no_grad():

    Z_train = model.encode(
        X_train_tensor
    )

    Z_val = model.encode(
        X_val_tensor
    )


# ============================================================
# CLASS WEIGHT
# ============================================================

normal_count = np.sum(
    y_train == 0
)


attack_count = np.sum(
    y_train == 1
)


# Give more importance to NORMAL because
# attack data is larger.

pos_weight_value = (

    normal_count
    /
    attack_count

)


print()
print(
    "Normal training samples:",
    normal_count
)

print(
    "Attack training samples:",
    attack_count
)

print(
    "Classifier positive weight:",
    pos_weight_value
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

).to(DEVICE)


# ============================================================
# CLASSIFIER DATASET
# ============================================================

y_train_tensor = torch.tensor(

    y_train,

    dtype=torch.float32

).unsqueeze(1).to(DEVICE)


classifier_dataset = TensorDataset(

    Z_train,

    y_train_tensor

)


classifier_loader = DataLoader(

    classifier_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True

)


# ============================================================
# CLASSIFIER LOSS
# ============================================================

pos_weight = torch.tensor(

    [pos_weight_value],

    dtype=torch.float32

).to(DEVICE)


classifier_loss = nn.BCEWithLogitsLoss(

    pos_weight=pos_weight

)


classifier_optimizer = torch.optim.Adam(

    classifier.parameters(),

    lr=LEARNING_RATE

)


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

print()
print("=" * 70)
print("TRAINING CLASSIFIER")
print("=" * 70)


classifier.train()


for epoch in range(EPOCHS):

    total_loss = 0.0


    for z_batch, y_batch in classifier_loader:

        classifier_optimizer.zero_grad()


        logits = classifier(
            z_batch
        )


        loss = classifier_loss(

            logits,

            y_batch

        )


        loss.backward()


        classifier_optimizer.step()


        total_loss += (

            loss.item()
            *
            len(z_batch)

        )


    average_loss = (

        total_loss
        /
        len(classifier_dataset)

    )


    print(

        f"Epoch "
        f"{epoch + 1:02d}/{EPOCHS} "
        f"| Loss: "
        f"{average_loss:.6f}"

    )


# ============================================================
# SAVE CLASSIFIER
# ============================================================

classifier_path = os.path.join(

    OUTPUT_DIR,

    "mobile_classifier.pth"

)


torch.save(

    classifier.state_dict(),

    classifier_path

)


print()
print(
    "Saved:",
    classifier_path
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 70)
print("VALIDATION")
print("=" * 70)


classifier.eval()


with torch.no_grad():

    logits = classifier(
        Z_val
    )


    probabilities = torch.sigmoid(
        logits
    ).cpu().numpy().flatten()


# ------------------------------------------------------------
# Probability decision
# ------------------------------------------------------------

prob_predictions = (

    probabilities
    >= ATTACK_PROBABILITY_THRESHOLD

).astype(int)


print()
print(
    f"Probability threshold: "
    f"{ATTACK_PROBABILITY_THRESHOLD}"
)


print()
print(
    classification_report(

        y_val,

        prob_predictions,

        target_names=[
            "NORMAL",
            "ATTACK"
        ],

        digits=4

    )
)


print()
print(
    "Confusion Matrix:"
)


print(

    confusion_matrix(

        y_val,

        prob_predictions

    )

)


print()
print(
    "Accuracy :",
    accuracy_score(
        y_val,
        prob_predictions
    )
)


print(
    "Precision:",
    precision_score(
        y_val,
        prob_predictions,
        zero_division=0
    )
)


print(
    "Recall   :",
    recall_score(
        y_val,
        prob_predictions,
        zero_division=0
    )
)


print(
    "F1 Score :",
    f1_score(
        y_val,
        prob_predictions,
        zero_division=0
    )
)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

features_path = os.path.join(

    OUTPUT_DIR,

    "mobile_features.pkl"

)


with open(

    features_path,

    "wb"

) as f:

    pickle.dump(

        FEATURE_COLUMNS,

        f

    )


print()
print(
    "Saved:",
    features_path
)


# ============================================================
# SAVE CONFIG
# ============================================================

config = {

    "feature_count": FEATURE_COUNT,

    "features": FEATURE_COLUMNS,

    "latent_size": LATENT_SIZE,

    "probability_threshold":
        ATTACK_PROBABILITY_THRESHOLD,

    "reconstruction_threshold":
        float(threshold),

    "threshold_k":
        THRESHOLD_K,

    "normal_samples":
        int(len(normal_df)),

    "attack_samples":
        int(len(attack_df))

}


config_path = os.path.join(

    OUTPUT_DIR,

    "mobile_config.json"

)


with open(

    config_path,

    "w"

) as f:

    json.dump(

        config,

        f,

        indent=4

    )


print(
    "Saved:",
    config_path
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("MOBILE MODEL TRAINING COMPLETE")
print("=" * 70)

print()
print("Generated files:")

print(
    "mobile_model_files/"
)

print(
    "├── mobile_model.pth"
)

print(
    "├── mobile_classifier.pth"
)

print(
    "├── mobile_scaler.pkl"
)

print(
    "├── mobile_threshold.pkl"
)

print(
    "├── mobile_features.pkl"
)

print(
    "└── mobile_config.json"
)

print()
print("DONE.")