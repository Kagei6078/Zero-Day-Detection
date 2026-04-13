import requests

url = "http://127.0.0.1:8000/predict"

# Sample 1: A basic array of 17 features (simulating normal traffic)
sample_normal = [120.0, 0.01, 1.0, 0.0, 0.0, 64.0, 64.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Sample 2: An array with massive numbers (simulating an anomaly/attack)
sample_attack = [99999.0, 50.0, 0.0, 1.0, 1.0, 8500.0, 12.0, 500.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

print("--- Testing Normal Sample ---")
response1 = requests.post(url, json={"sample": sample_normal})
print(response1.json())

print("\n--- Testing Attack Sample ---")
response2 = requests.post(url, json={"sample": sample_attack})
print(response2.json())