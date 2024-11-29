# Import library yang diperlukan
import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Muat model yang sudah disimpan
model = joblib.load('heart_attack_model.pkl')
scaler = joblib.load('scaler.pkl')

# Fungsi untuk memprediksi risiko serangan jantung dalam skala 1-10
def predict_heart_attack(data):
    # Nama fitur yang digunakan oleh model
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                     'thalachh', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    # Mengubah data input menjadi DataFrame dengan nama kolom yang sesuai
    data_df = pd.DataFrame([data], columns=feature_names)
    
    # Standarisasi data
    data_scaled = scaler.transform(data_df)
    prediction = model.predict_proba(data_scaled)[0]  # Dapatkan probabilitas untuk kedua kelas
    risk_score = prediction[1] * 10  # Mengambil probabilitas untuk kelas 'serangan jantung' (kelas 1)
    return round(risk_score)  # Kembalikan hasil sebagai angka antara 1-10

# Fungsi untuk memvisualisasikan data
def plot_data(df):
    fig = px.scatter(df, x='age', y='chol', color='target', 
                     title="Hubungan Usia dan Kolesterol terhadap Risiko Serangan Jantung",
                     labels={'age': 'Usia', 'chol': 'Kadar Kolesterol (mg/dl)', 'target': 'Risiko'})
    st.plotly_chart(fig)

    # Tambahkan grafik lainnya jika diinginkan
    fig2 = go.Figure(data=[go.Histogram(x=df['age'], nbinsx=20)])
    fig2.update_layout(title="Distribusi Usia", xaxis_title="Usia", yaxis_title="Jumlah")
    st.plotly_chart(fig2)

# Membuat halaman multipage
st.title("Aplikasi Deteksi Serangan Jantung")
pages = ["Prediksi Risiko Serangan Jantung", "Visualisasi Data"]
page = st.sidebar.radio("Pilih Halaman", pages)

if page == "Prediksi Risiko Serangan Jantung":
    # Input fitur dari pengguna
    age = st.number_input("Usia (tahun)", min_value=0, max_value=100, value=50)
    sex = st.selectbox("Jenis Kelamin", options=[0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")
    cp = st.selectbox("Tipe Nyeri Dada", options=[0, 1, 2, 3], format_func=lambda x: {0: "Angina Tipikal", 1: "Angina Atypical", 2: "Nyeri Non-Anginal", 3: "Asimptomatik"}[x])
    trestbps = st.number_input("Tekanan Darah Saat Istirahat (mm Hg)", min_value=50, max_value=200, value=120)
    chol = st.number_input("Kadar Kolesterol (mg/dl)", min_value=100, max_value=400, value=200)
    fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl", options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    restecg = st.selectbox("Hasil EKG Saat Istirahat", options=[0, 1, 2], format_func=lambda x: {0: "Normal", 1: "Abnormalitas Gelombang ST-T", 2: "Hipertrofi Ventrikel Kiri"}[x])
    thalachh = st.number_input("Detak Jantung Maksimum Yang Dicapai", min_value=50, max_value=220, value=150)
    exang = st.selectbox("Angina Induksi Latihan", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
    oldpeak = st.number_input("Depresi ST Saat Latihan Dibandingkan Dengan Istirahat", min_value=0.0, max_value=6.0, value=1.0)
    slope = st.selectbox("Kemiringan Segmen ST Puncak Latihan", options=[0, 1, 2], format_func=lambda x: {0: "Meningkat", 1: "Datar", 2: "Menurun"}[x])
    ca = st.selectbox("Jumlah Pembuluh Darah Utama (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox("Jenis Thalassemia", options=[1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Defek Tetap", 3: "Defek Reversibel"}[x])

    # Mengambil input fitur dalam bentuk list
    data_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalachh, exang, oldpeak, slope, ca, thal]

    # Prediksi jika tombol ditekan
    if st.button("Prediksi Risiko Serangan Jantung"):
        risk_score = predict_heart_attack(data_input)

        # Tampilkan hasil prediksi
        st.write(f"Skor Risiko Serangan Jantung: {risk_score} (Skala 1-10)")

        if risk_score >= 8:
            st.warning("Risiko Sangat Tinggi: Pasien berisiko sangat besar mengalami serangan jantung.")
        elif risk_score >= 5:
            st.warning("Risiko Sedang: Pasien memiliki risiko sedang mengalami serangan jantung.")
        else:
            st.success("Risiko Rendah: Pasien memiliki peluang lebih kecil mengalami serangan jantung.")

elif page == "Visualisasi Data":
    # Fitur untuk visualisasi data
    st.write("Silakan unggah file CSV untuk melihat visualisasi data.")
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        plot_data(df)
