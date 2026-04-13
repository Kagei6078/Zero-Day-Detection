import pandas as pd
import requests

print("Loading real data from your CSV...")

# 1. Load the dataset
df = pd.read_csv("Network_dataset_2.csv", low_memory=False)
df['type'] = df['type'].astype(str).str.lower()

# 2. Separate normal and attack data
normal_df = df[df['type'] == 'normal']
attack_df = df[df['type'] == 'scanning']

# 3. Grab the exact 17 numeric features your model expects
numeric_cols = normal_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
features_to_use = numeric_cols[:17]

# 4. Extract the very first row of each as a list of floats
real_normal_sample = normal_df[features_to_use].iloc[0].fillna(0).tolist()
real_attack_sample = attack_df[features_to_use].iloc[0].fillna(0).tolist()

# 5. Send to your FastAPI server
url = "http://127.0.0.1:8000/predict"

print("\n--- Testing REAL Normal Sample ---")
# Expected Output: NORMAL (Low Error, Low Probability)
response1 = requests.post(url, json={"sample": real_normal_sample})
print(response1.json())

print("\n--- Testing REAL Attack (Scanning) Sample ---")
# Expected Output: ATTACK (High Probability from the Classifier)
response2 = requests.post(url, json={"sample": real_attack_sample})
print(response2.json())