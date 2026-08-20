import numpy as np
from sklearn.ensemble import IsolationForest

# --- 1. Load / Prepare Training Data ---
# Representing your 2000 vectors of dimension N (e.g., N = 128)
N = 128
X_train = np.random.randn(2000, N)  # Clean training data

# --- 2. Initialize and Train the Model ---
# contamination='auto' prevents forced flagging of training samples as outliers.
# max_samples=256 is recommended by the authors for path length stability.
model = IsolationForest(
    n_estimators=100,
    max_samples=min(256, len(X_train)),
    contamination='auto',
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

model.fit(X_train)

# --- 3. Evaluate New / Unseen Vectors ---
# Suppose you receive 5 new vectors to test
X_new = np.random.randn(5, N)
# Add an intentional anomaly (e.g., extreme values)
X_new[0] = X_new[0] + 10.0

# Predict binary labels: +1 (Normal), -1 (Anomaly)
predictions = model.predict(X_new)

# Get raw anomaly scores (more negative = more anomalous)
scores = model.score_samples(X_new)

for i, (pred, score) in enumerate(zip(predictions, scores)):
    status = "ANOMALY" if pred == -1 else "Normal"
    print(f"Vector {i}: Status={status:<7} | Score={score:.4f}")