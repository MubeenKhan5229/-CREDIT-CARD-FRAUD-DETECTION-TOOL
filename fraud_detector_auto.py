import subprocess
import sys
import os

# =========================
# AUTO INSTALL
# =========================
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import pandas
    import numpy
    import sklearn
except ImportError:
    install("pandas")
    install("numpy")
    install("scikit-learn")

# =========================
# IMPORTS
# =========================
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"
DATASET_FILE = "creditcard.csv"

# =========================
# TRAIN MODEL (AUTO)
# =========================
def train_model():
    if not os.path.exists(DATASET_FILE):
        messagebox.showerror("Error", "creditcard.csv not found!")
        sys.exit()

    df = pd.read_csv(DATASET_FILE)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    pickle.dump(model, open(MODEL_FILE, "wb"))
    pickle.dump(scaler, open(SCALER_FILE, "wb"))

    return model, scaler

# =========================
# LOAD OR TRAIN
# =========================
if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
    model = pickle.load(open(MODEL_FILE, "rb"))
    scaler = pickle.load(open(SCALER_FILE, "rb"))
else:
    model, scaler = train_model()

# =========================
# GUI
# =========================
root = tk.Tk()
root.title("Fraud Detection System")
root.geometry("700x450")
root.configure(bg="black")

def predict():
    try:
        values = entry.get().split(",")
        values = [float(i.strip()) for i in values]
        values = np.array(values).reshape(1, -1)
        values = scaler.transform(values)

        prediction = model.predict(values)[0]

        if prediction == 1:
            result.config(text="FRAUD DETECTED", fg="red")
        else:
            result.config(text="LEGITIMATE TRANSACTION", fg="green")

    except:
        messagebox.showerror("Error", "Invalid Input")

title = tk.Label(root, text="Credit Card Fraud Detection",
                 font=("Arial", 18), bg="black", fg="white")
title.pack(pady=20)

entry = tk.Entry(root, width=80)
entry.pack(pady=10)

btn = tk.Button(root, text="Predict",
                command=predict, bg="green", fg="white")
btn.pack(pady=10)

result = tk.Label(root, text="", font=("Arial", 16),
                  bg="black")
result.pack(pady=20)

root.mainloop()
