import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
import numpy as np

# === Environment Check ===
print("Python Interpreter:", sys.executable)
print("Seaborn version:", sns.__version__)
print("Pandas version:", pd.__version__)

# === Load Dataset ===
print("\nLoading dataset...")
df = pd.read_csv("data/creditcard.csv")

# === Preprocessing ===
print("\nPreprocessing data...")
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df = df.drop(['Time'], axis=1)

# === Isolation Forest ===
print("\nTraining Isolation Forest...")
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(df.drop('Class', axis=1))

print("\nDetecting anomalies (Isolation Forest)...")
df['anomaly'] = model.predict(df.drop('Class', axis=1))
df['anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)

print("\nConfusion Matrix (Isolation Forest):")
print(confusion_matrix(df['Class'], df['anomaly']))
print("\nClassification Report (Isolation Forest):")
print(classification_report(df['Class'], df['anomaly'], digits=4))

df[df['anomaly'] == 1].to_csv("anomaly_results.csv", index=False)
print("\nAnomalies (Isolation Forest) saved to 'anomaly_results.csv'")

# === Visualize Isolation Forest Result ===
sns.countplot(x='anomaly', data=df)
plt.title("Isolation Forest - Detected Anomalies")
plt.xlabel("Anomaly (1 = Outlier)")
plt.ylabel("Count")
plt.show()

# === Autoencoder ===
print("\n\n🔁 Starting Autoencoder-based Anomaly Detection...")

X = df[df['Class'] == 0].drop(['Class', 'anomaly'], axis=1)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoder = Dense(16, activation='relu')(input_layer)
encoder = Dense(8, activation='relu')(encoder)
decoder = Dense(16, activation='relu')(encoder)
decoder = Dense(input_dim, activation='linear')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(X_train, X_train, epochs=10, batch_size=64, validation_data=(X_test, X_test), verbose=1)

X_all = df.drop(['Class', 'anomaly'], axis=1)
reconstructions = autoencoder.predict(X_all)

mse = np.mean(np.power(X_all - reconstructions, 2), axis=1)
threshold = np.percentile(mse, 99)
df['autoencoder_error'] = mse
df['autoencoder_anomaly'] = df['autoencoder_error'] > threshold
df['autoencoder_anomaly'] = df['autoencoder_anomaly'].astype(int)

print("\nConfusion Matrix (Autoencoder):")
print(confusion_matrix(df['Class'], df['autoencoder_anomaly']))
print("\nClassification Report (Autoencoder):")
print(classification_report(df['Class'], df['autoencoder_anomaly'], digits=4))

df[df['autoencoder_anomaly'] == 1].to_csv("autoencoder_anomaly_results.csv", index=False)
print("\nAnomalies (Autoencoder) saved to 'autoencoder_anomaly_results.csv'")

sns.histplot(df['autoencoder_error'], bins=100, kde=True)
plt.axvline(threshold, color='red', linestyle='--', label='Anomaly Threshold')
plt.title("Autoencoder Reconstruction Error")
plt.xlabel("Error")
plt.ylabel("Count")
plt.legend()
plt.show()

input("\nPress Enter to exit...")
