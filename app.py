import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Anomaly Detection", layout="wide")
st.title("🔍 Anomaly Detection in Financial Transactions")

uploaded_file = st.file_uploader("📂 Upload your CSV file (e.g., creditcard.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    if 'Amount' in df.columns:
        if 'Time' in df.columns:
            df.drop(['Time'], axis=1, inplace=True)

        df['Amount'] = StandardScaler().fit_transform(df[['Amount']])

        # Calculate true anomaly rate (capped at 20%)
        true_anomaly_ratio = df['Class'].mean()
        if true_anomaly_ratio > 0.2:
            st.warning(f"⚠️ High anomaly rate detected: {true_anomaly_ratio:.2%}. Capping threshold at 20%.")
            true_anomaly_ratio = 0.2

        model_option = st.radio("🤖 Choose model to apply:", ["Isolation Forest", "Autoencoder", "Both"], horizontal=True)

        if st.button("🚀 Run Detection"):
            report_buffer = io.StringIO()

            if model_option in ["Isolation Forest", "Both"]:
                st.subheader("🔍 Isolation Forest")
                iso_model = IsolationForest(n_estimators=200, contamination=true_anomaly_ratio, random_state=42)
                iso_model.fit(df.drop('Class', axis=1))
                df['anomaly_if'] = iso_model.predict(df.drop('Class', axis=1))
                df['anomaly_if'] = df['anomaly_if'].apply(lambda x: 1 if x == -1 else 0)

                cm_if = confusion_matrix(df['Class'], df['anomaly_if'])
                report_if = classification_report(df['Class'], df['anomaly_if'], output_dict=True, digits=4)
                st.dataframe(pd.DataFrame(cm_if, columns=["Predicted Normal", "Predicted Anomaly"], index=["Actual Normal", "Actual Fraud"]))
                st.dataframe(pd.DataFrame(report_if).T)

                # Bar Chart
                st.subheader("📊 IF Bar Chart")
                fig_bar_if, ax_bar_if = plt.subplots()
                df['anomaly_if'].value_counts().sort_index().plot(kind='bar', ax=ax_bar_if)
                ax_bar_if.set_xticks([0, 1])
                ax_bar_if.set_xticklabels(['Normal', 'Anomaly'])
                ax_bar_if.set_ylabel('Count')
                st.pyplot(fig_bar_if)

                # Pie Chart
                st.subheader("🥧 IF Pie Chart")
                fig_pie_if, ax_pie_if = plt.subplots()
                df['anomaly_if'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Normal', 'Anomaly'], ax=ax_pie_if)
                ax_pie_if.axis('equal')
                st.pyplot(fig_pie_if)

                report_buffer.write("Isolation Forest Classification Report:\n")
                report_buffer.write(pd.DataFrame(report_if).T.to_string())
                report_buffer.write("\n\n")

            if model_option in ["Autoencoder", "Both"]:
                st.subheader("🔁 Autoencoder")
                X = df[df['Class'] == 0].drop(['Class'], axis=1)
                if 'anomaly_if' in X.columns:
                    X = X.drop(['anomaly_if'], axis=1)
                X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

                input_dim = X_train.shape[1]
                input_layer = Input(shape=(input_dim,))
                encoded = Dense(32, activation='relu')(input_layer)
                encoded = Dense(16, activation='relu')(encoded)
                decoded = Dense(32, activation='relu')(encoded)
                decoded = Dense(input_dim, activation='linear')(decoded)

                autoencoder = Model(inputs=input_layer, outputs=decoded)
                autoencoder.compile(optimizer='adam', loss='mse')
                autoencoder.fit(X_train, X_train, epochs=20, batch_size=64, validation_data=(X_test, X_test), verbose=0)

                X_all = df.drop(['Class'], axis=1)
                if 'anomaly_if' in X_all.columns:
                    X_all = X_all.drop(['anomaly_if'], axis=1)
                reconstructions = autoencoder.predict(X_all)
                mse = np.mean(np.power(X_all - reconstructions, 2), axis=1)
                ae_threshold = np.percentile(mse, 100 - (true_anomaly_ratio * 100))
                df['ae_error'] = mse
                df['ae_anomaly'] = (df['ae_error'] > ae_threshold).astype(int)

                cm_ae = confusion_matrix(df['Class'], df['ae_anomaly'])
                report_ae = classification_report(df['Class'], df['ae_anomaly'], output_dict=True, digits=4)
                st.dataframe(pd.DataFrame(cm_ae, columns=["Predicted Normal", "Predicted Anomaly"], index=["Actual Normal", "Actual Fraud"]))
                st.dataframe(pd.DataFrame(report_ae).T)

                # Bar Chart
                st.subheader("📊 AE Bar Chart")
                fig_bar_ae, ax_bar_ae = plt.subplots()
                df['ae_anomaly'].value_counts().sort_index().plot(kind='bar', ax=ax_bar_ae)
                ax_bar_ae.set_xticks([0, 1])
                ax_bar_ae.set_xticklabels(['Normal', 'Anomaly'])
                ax_bar_ae.set_ylabel('Count')
                st.pyplot(fig_bar_ae)

                # Pie Chart
                st.subheader("🥧 AE Pie Chart")
                fig_pie_ae, ax_pie_ae = plt.subplots()
                df['ae_anomaly'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Normal', 'Anomaly'], ax=ax_pie_ae)
                ax_pie_ae.axis('equal')
                st.pyplot(fig_pie_ae)

                report_buffer.write("Autoencoder Classification Report:\n")
                report_buffer.write(pd.DataFrame(report_ae).T.to_string())

            st.download_button("📄 Download Classification Report", report_buffer.getvalue(), file_name="classification_report.txt")
            st.success("✅ Detection completed!")
