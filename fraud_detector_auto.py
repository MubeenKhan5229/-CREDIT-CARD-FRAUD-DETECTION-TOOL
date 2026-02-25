import subprocess
import sys
import os

# =========================
# AUTO INSTALL FUNCTION
# =========================
def auto_install():
    required_packages = ["pandas", "numpy", "scikit-learn"]

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

auto_install()

# =========================
# IMPORT AFTER INSTALL
# =========================
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


MODEL_FILE = "fraud_model.pkl"
SCALER_FILE = "scaler.pkl"
DATASET_FILE = "creditcard.csv"


# =========================
# TRAIN MODEL
# =========================
def train_model():
    if not os.path.exists(DATASET_FILE):
        messagebox.showerror("Error", "Dataset 'creditcard.csv' not found!")
        return None, None

    print("Training model... Please wait.")
    df = pd.read_csv(DATASET_FILE)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print("Model trained. ROC-AUC:", auc)

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

if model is None:
    sys.exit()


# =========================
# PREDICT SINGLE
# =========================
def predict_single():
    try:
        values = entry.get().split(",")
        values = [float(v.strip()) for v in values]

        values = np.array(values).reshape(1, -1)
        values_scaled = scaler.transform(values)

        prediction = model.predict(values_scaled)[0]
        probability = model.predict_proba(values_scaled)[0][1]

        if prediction == 1:
            result_label.config(
                text=f"⚠ FRAUD DETECTED\nProbability: {probability:.4f}",
                fg="red"
            )
        else:
            result_label.config(
                text=f"✅ Legitimate Transaction\nProbability: {probability:.4f}",
                fg="green"
            )
    except Exception as e:
        messagebox.showerror("Error", f"Invalid Input!\n{str(e)}")


# =========================
# PREDICT CSV
# =========================
def predict_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            data = pd.read_csv(file_path)
            data_scaled = scaler.transform(data)

            predictions = model.predict(data_scaled)
            probabilities = model.predict_proba(data_scaled)[:, 1]

            data["Prediction"] = predictions
            data["Fraud_Probability"] = probabilities

            output_file = "predictions_output.csv"
            data.to_csv(output_file, index=False)

            messagebox.showinfo("Success", f"Saved as {output_file}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


# =========================
# GUI
# =========================
root = tk.Tk()
root.title("Credit Card Fraud Detection System")
root.geometry("750x500")
root.configure(bg="#121212")

title = tk.Label(root,
                 text="💳 Credit Card Fraud Detection",
                 font=("Arial", 22, "bold"),
                 bg="#121212",
                 fg="white")
title.pack(pady=20)

info = tk.Label(root,
                text="Enter ALL feature values (comma separated)",
                bg="#121212",
                fg="lightgray")
info.pack()

entry = tk.Entry(root, width=95)
entry.pack(pady=15)

predict_btn = tk.Button(root,
                        text="Predict Single Transaction",
                        command=predict_single,
                        bg="#4CAF50",
                        fg="white",
                        width=30,
                        height=2)
predict_btn.pack(pady=10)

csv_btn = tk.Button(root,
                    text="Predict CSV File",
                    command=predict_csv,
                    bg="#2196F3",
                    fg="white",
                    width=30,
                    height=2)
csv_btn.pack(pady=10)

result_label = tk.Label(root,
                        text="",
                        font=("Arial", 16, "bold"),
                        bg="#121212")
result_label.pack(pady=30)

footer = tk.Label(root,
                  text="Auto Install | Random Forest | Imbalanced Data Handled",
                  bg="#121212",
                  fg="gray")
footer.pack(side="bottom", pady=10)

root.mainloop()
