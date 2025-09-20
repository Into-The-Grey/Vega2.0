#!/usr/bin/env python3
"""
Debug script to understand the anomaly detection behavior
"""

import sys

sys.path.insert(0, "/home/ncacord/Vega2.0/src/vega/federated")

from security import is_anomalous_update
import json

# Test the function step by step
print("🔍 Debugging Anomaly Detection")

# Test data that should be anomalous
anomalous_update = {
    "weights": {
        "layer1": [100.0, 2.0, 3.0],  # Large weight
        "layer2": [0.5, -0.3, 0.8],
    },
    "metadata": {"epoch": 5},
}

print("Input data:")
print(json.dumps(anomalous_update, indent=2))

# Test with threshold 10.0
result = is_anomalous_update(
    anomalous_update,
    threshold=10.0,
    participant_id="test_participant",
    session_id="test_session",
)

print("\nAnomaly detection result:")
print(json.dumps(result, indent=2))

# Also test with flattened data
flattened_update = {
    "layer1_weight_0": 100.0,  # Direct large value
    "layer1_weight_1": 2.0,
    "layer2_weight_0": 0.5,
}

print("\nTesting with flattened data:")
print(json.dumps(flattened_update, indent=2))

result2 = is_anomalous_update(
    flattened_update,
    threshold=10.0,
    participant_id="test_participant",
    session_id="test_session",
)

print("\nFlattened result:")
print(json.dumps(result2, indent=2))
