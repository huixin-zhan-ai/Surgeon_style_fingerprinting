import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
from xgboost import XGBClassifier  # Make sure xgboost is installed

# Generate fake data (your existing embeddings setup)
HIDDEN_DIM = 512
num_fake = 50
embeddings_in_train = embeddings
embeddings_not_in_train = np.random.normal(loc=0.0, scale=1.0, size=(num_fake, HIDDEN_DIM))

# Combine data
X_all = np.vstack([embeddings_in_train, embeddings_not_in_train])
y_all = np.array([1] * len(embeddings_in_train) + [0] * num_fake)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
)

# 🔁 Use XGBoost for attack model
clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
clf.fit(X_train, y_train)

# Predict membership probability
y_pred = clf.predict_proba(X_test)[:, 1]
y_bin_pred = (y_pred >= 0.5).astype(int)

# Compute metrics
acc = accuracy_score(y_test, y_bin_pred)
precision = precision_score(y_test, y_bin_pred)
recall = recall_score(y_test, y_bin_pred)
f1 = f1_score(y_test, y_bin_pred)
auc = roc_auc_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_bin_pred)

# Print results
print("Membership Inference Metrics (XGBoost):")
print(f"  AUC-ROC     : {auc:.3f}")
print(f"  Accuracy    : {acc:.3f}")
print(f"  Precision   : {precision:.3f}")
print(f"  Recall      : {recall:.3f}")
print(f"  F1 Score    : {f1:.3f}")
print(f"  Confusion Matrix:\n{cm}")