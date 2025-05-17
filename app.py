import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Anomaly Detection", layout="wide")
st.markdown("<h1 style='text-align: center;'>🔍 Anomaly Detection in Financial Transactions</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload your CSV file (e.g., creditcard.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    if 'Amount' in df.columns and 'Time' in df.columns:
        df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
        df.drop(['Time'], axis=1, inplace=True)

        model_option = st.radio("🤖 Choose model to apply:", ["Isolation Forest", "Autoencoder", "Both"], horizontal=True)

        if st.button("🚀 Run Detection"):
            if model_option in ["Isolation Forest", "Both"]:
                st.subheader("🔍 Isolation Forest")
                iso_model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
                iso_model.fit(df.drop('Class', axis=1))
                df['anomaly_if'] = iso_model.predict(df.drop('Class', axis=1))
                df['anomaly_if'] = df['anomaly_if'].apply(lambda x: 1 if x == -1 else 0)

                cm = confusion_matrix(df['Class'], df['anomaly_if'])
                st.text("Confusion Matrix (IF):")
                st.text(cm)
                st.text("Classification Report (IF):")
                st.text(classification_report(df['Class'], df['anomaly_if'], digits=4))

                # Bar chart
                st.subheader("📊 IF Bar Chart")
                counts = pd.Series(df['anomaly_if']).value_counts().sort_index()
                fig1, ax1 = plt.subplots()
                sns.barplot(x=counts.index.map({0: 'Normal', 1: 'Anomaly'}), y=counts.values, ax=ax1)
                ax1.set_ylabel("Count")
                st.pyplot(fig1)

                # Pie chart
                st.subheader("🥧 IF Pie Chart")
                fig2, ax2 = plt.subplots()
                ax2.pie(counts.values, labels=['Normal', 'Anomaly'], autopct='%1.1f%%', startangle=90)
                ax2.axis('equal')
                st.pyplot(fig2)

                anomalies_if = df[df['anomaly_if'] == 1]
                st.download_button("📥 Download IF Anomalies (CSV)", anomalies_if.to_csv(index=False).encode('utf-8'), "if_anomalies.csv")

            if model_option in ["Autoencoder", "Both"]:
                st.subheader("🔁 Autoencoder")
                X = df[df['Class'] == 0].drop(['Class'], axis=1)
                if 'anomaly_if' in X.columns:
                    X = X.drop(['anomaly_if'], axis=1)
                X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

                input_dim = X_train.shape[1]
                input_layer = Input(shape=(input_dim,))
                encoder = Dense(16, activation='relu')(input_layer)
                encoder = Dense(8, activation='relu')(encoder)
                decoder = Dense(16, activation='relu')(encoder)
                decoder = Dense(input_dim, activation='linear')(decoder)
                autoencoder = Model(inputs=input_layer, outputs=decoder)
                autoencoder.compile(optimizer='adam', loss='mse')
                autoencoder.fit(X_train, X_train, epochs=10, batch_size=64, validation_data=(X_test, X_test), verbose=0)

                X_all = df.drop(['Class'], axis=1)
                if 'anomaly_if' in X_all.columns:
                    X_all = X_all.drop(['anomaly_if'], axis=1)

                reconstructions = autoencoder.predict(X_all)
                mse = np.mean(np.power(X_all - reconstructions, 2), axis=1)
                threshold = np.percentile(mse, 99)
                df['ae_error'] = mse
                df['ae_anomaly'] = (mse > threshold).astype(int)

                st.text("Confusion Matrix (AE):")
                st.text(confusion_matrix(df['Class'], df['ae_anomaly']))
                st.text("Classification Report (AE):")
                st.text(classification_report(df['Class'], df['ae_anomaly'], digits=4))

                # Bar chart
                st.subheader("📊 AE Bar Chart")
                counts_ae = pd.Series(df['ae_anomaly']).value_counts().sort_index()
                fig3, ax3 = plt.subplots()
                sns.barplot(x=counts_ae.index.map({0: 'Normal', 1: 'Anomaly'}), y=counts_ae.values, ax=ax3)
                ax3.set_ylabel("Count")
                st.pyplot(fig3)

                # Pie chart
                st.subheader("🥧 AE Pie Chart")
                fig4, ax4 = plt.subplots()
                ax4.pie(counts_ae.values, labels=['Normal', 'Anomaly'], autopct='%1.1f%%', startangle=90)
                ax4.axis('equal')
                st.pyplot(fig4)

                anomalies_ae = df[df['ae_anomaly'] == 1]
                st.download_button("📥 Download AE Anomalies (CSV)", anomalies_ae.to_csv(index=False).encode('utf-8'), "ae_anomalies.csv")
