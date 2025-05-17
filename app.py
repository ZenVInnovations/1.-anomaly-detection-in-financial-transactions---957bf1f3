import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Title
st.title("🔍 Anomaly Detection in Financial Transactions")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    if 'Amount' in df.columns and 'Time' in df.columns:
        # Preprocessing
        df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
        df.drop(['Time'], axis=1, inplace=True)

        model_option = st.selectbox("Select Model", ("Isolation Forest", "Autoencoder"))

        if st.button("Detect Anomalies"):
            with st.spinner("Running..."):

                if model_option == "Isolation Forest":
                    iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
                    iso.fit(df.drop('Class', axis=1))
                    df['anomaly'] = iso.predict(df.drop('Class', axis=1))
                    df['anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)

                else:
                    # Autoencoder logic
                    X = df[df['Class'] == 0].drop(['Class'], axis=1)
                    X_train = X.sample(frac=0.8, random_state=42)
                    X_test = X.drop(X_train.index)

                    input_dim = X_train.shape[1]
                    input_layer = Input(shape=(input_dim,))
                    encoder = Dense(16, activation='relu')(input_layer)
                    encoder = Dense(8, activation='relu')(encoder)
                    decoder = Dense(16, activation='relu')(encoder)
                    decoder = Dense(input_dim, activation='linear')(decoder)
                    autoencoder = Model(inputs=input_layer, outputs=decoder)
                    autoencoder.compile(optimizer='adam', loss='mse')
                    autoencoder.fit(X_train, X_train, epochs=10, batch_size=64, validation_data=(X_test, X_test), verbose=0)

                    reconstructions = autoencoder.predict(df.drop('Class', axis=1))
                    mse = np.mean(np.power(df.drop('Class', axis=1) - reconstructions, 2), axis=1)
                    threshold = np.percentile(mse, 99)
                    df['anomaly'] = (mse > threshold).astype(int)

            st.subheader("📊 Anomaly Detection Result")
            st.write(df[['Class', 'anomaly']].value_counts().rename("Count"))

            st.download_button("📥 Download Anomalies", data=df[df['anomaly'] == 1].to_csv(index=False),
                               file_name="anomalies.csv", mime="text/csv")
