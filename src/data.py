
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_diabetes():
    data = fetch_openml("diabetes", version=1, as_frame=True)

    X = data.data.replace(0, np.nan)
    X = X.fillna(X.median())

    y = (data.target == "tested_positive").astype(int).values

    return X.values, y, X.columns.tolist()

def prepare_data(test_size=0.2):
    X, y, features = load_diabetes()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    print("=== DATASET SUMMARY ===")
    print("Total samples:", len(y))
    print("Positive (Diabetes):", (y == 1).sum())
    print("Negative (Non-Diabetes):", (y == 0).sum())
    print("Features:", features)
    print("="*50)
    print()

    return Xtr, Xte, ytr, yte, features


