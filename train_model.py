
import os
import pandas as pd
import numpy as np
import joblib

from urllib.parse import urlparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from feature_extraction import extract_features


# --------------------------------
# 1. LOAD AND CLEAN DATASET
# --------------------------------

DATASET_PATH = "dataset/urls.csv"

data = pd.read_csv(DATASET_PATH)

if not {"url", "verdict"}.issubset(data.columns):
    raise ValueError("Dataset must contain 'url' and 'verdict' columns.")

data = data.dropna(subset=["url", "verdict"]).copy()

# Remove exact duplicate URLs
data = data.drop_duplicates(subset=["url"]).reset_index(drop=True)

data["verdict"] = data["verdict"].astype(int)


# --------------------------------
# 2. EXTRACT HOSTNAMES
# --------------------------------

def get_hostname(url):
    url = str(url).strip()

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    return urlparse(url).hostname or ""


data["hostname"] = data["url"].apply(get_hostname)

# Remove records with no usable hostname
data = data[data["hostname"] != ""].reset_index(drop=True)


# --------------------------------
# 3. EXTRACT URL FEATURES
# --------------------------------

X = pd.DataFrame(
    [extract_features(url) for url in data["url"]]
)

y = data["verdict"]
groups = data["hostname"]

print("\n========== DATASET SUMMARY ==========")
print("Total URLs:", len(data))
print("Unique hostnames:", groups.nunique())
print("\nClass distribution:")
print(y.value_counts())


# --------------------------------
# 4. GROUP-AWARE EVALUATION
# --------------------------------

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_predictions = np.zeros(len(data), dtype=int)

print("\n========== UNSEEN-HOSTNAME TEST ==========")

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups), start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_hosts = set(groups.iloc[train_idx])
    test_hosts = set(groups.iloc[test_idx])

    overlap = train_hosts.intersection(test_hosts)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    oof_predictions[test_idx] = predictions

    print(f"\nFold {fold}")
    print("Training URLs:", len(train_idx))
    print("Testing URLs:", len(test_idx))
    print("Hostname overlap:", len(overlap))
    print("Accuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))


# --------------------------------
# 5. OVERALL EVALUATION
# --------------------------------

print("\n========== OVERALL RESULTS ==========")

print("\nAccuracy:")
print(accuracy_score(y, oof_predictions))

print("\nClassification Report:")
print(classification_report(
    y,
    oof_predictions,
    target_names=["Legitimate", "Phishing"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y, oof_predictions))


# --------------------------------
# 6. TRAIN FINAL MODEL
# --------------------------------

final_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

final_model.fit(X, y)


# --------------------------------
# 7. SAVE MODEL
# --------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(final_model, "models/phishing_model.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")

print("\nFinal model trained on the full dataset.")
print("Model saved successfully!")
