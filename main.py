import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# === Load Dataset ===
df = pd.read_csv("data/creditcard.csv")

# === Preprocessing ===
if 'Time' in df.columns:
    df.drop(['Time'], axis=1, inplace=True)

scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# === Determine True Anomaly Rate (and cap it at 20%) ===
true_anomaly_ratio = df['Class'].mean()
if true_anomaly_ratio > 0.2:
    print(f"⚠️ High anomaly rate detected: {true_anomaly_ratio:.2%}. Capping threshold at 20%.")
    true_anomaly_ratio = 0.2

# === Isolation Forest ===
print("\n[Isolation Forest] Training...")
iso_model = IsolationForest(n_estimators=200, contamination=true_anomaly_ratio, random_state=42)
iso_model.fit(df.drop('Class', axis=1))

print("[Isolation Forest] Predicting anomalies...")
df['anomaly_if'] = iso_model.predict(df.drop('Class', axis=1))
df['anomaly_if'] = df['anomaly_if'].apply(lambda x: 1 if x == -1 else 0)

# === Isolation Forest Results ===
print("\n[Isolation Forest] Confusion Matrix:")
cm_if = confusion_matrix(df['Class'], df['anomaly_if'])
print(pd.DataFrame(cm_if, columns=["Predicted Normal", "Predicted Anomaly"],
                   index=["Actual Normal", "Actual Fraud"]))

print("\n[Isolation Forest] Classification Report:")
print(pd.DataFrame(classification_report(df['Class'], df['anomaly_if'], digits=4, output_dict=True)).T)

# === Autoencoder ===
print("\n[Autoencoder] Training...")
X = df[df['Class'] == 0].drop(['Class', 'anomaly_if'], axis=1)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=20, batch_size=64, validation_data=(X_test, X_test), verbose=1)

print("[Autoencoder] Predicting reconstruction errors...")
X_all = df.drop(['Class', 'anomaly_if'], axis=1)
reconstructions = autoencoder.predict(X_all)
mse = np.mean(np.power(X_all - reconstructions, 2), axis=1)

threshold = np.percentile(mse, 100 - (true_anomaly_ratio * 100))
print(f"[Autoencoder] Anomaly threshold ({100 - (true_anomaly_ratio * 100):.2f} percentile): {threshold:.6f}")

df['ae_error'] = mse
df['ae_anomaly'] = (df['ae_error'] > threshold).astype(int)

# === Autoencoder Results ===
print("\n[Autoencoder] Confusion Matrix:")
cm_ae = confusion_matrix(df['Class'], df['ae_anomaly'])
print(pd.DataFrame(cm_ae, columns=["Predicted Normal", "Predicted Anomaly"],
                   index=["Actual Normal", "Actual Fraud"]))

print("\n[Autoencoder] Classification Report:")
print(pd.DataFrame(classification_report(df['Class'], df['ae_anomaly'], digits=4, output_dict=True)).T)

# === Save Results ===
df.to_csv("full_anomaly_results.csv", index=False)
print("\n✅ Results saved to 'full_anomaly_results.csv'")
