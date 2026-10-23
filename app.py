import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Config Tampilan Aplikasi
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="🛍️",
    layout="wide"
)

# Load Model dan Scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load('model_kmeans.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Gagal memuat model/scaler. Pastikan file .pkl sudah ada! Error: {e}")
    st.stop()

# Header Aplikasi
st.title("🛍️ Sistem Segmentasi Pelanggan E-Commerce")
st.markdown("""
Aplikasi ini memanfaatkan model **K-Means Clustering** untuk mengelompokkan pelanggan berdasarkan karakteristik keaktifan, masa berlangganan, dan nominal *cashback*.
""")

st.divider()

# Form Input Data Pelanggan Baru
st.subheader("📋 Input Profil Pelanggan Baru")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Masa Keanggotaan / Tenure (Bulan)", min_value=0.0, max_value=100.0, value=12.0)
    warehouse_to_home = st.number_input("Jarak Gudang ke Rumah (KM)", min_value=0.0, max_value=200.0, value=15.0)
    hour_spend = st.number_input("Waktu Akses Aplikasi (Jam/Bulan)", min_value=0.0, max_value=24.0, value=3.0)

with col2:
    order_count = st.number_input("Total Pesanan Bulan Lalu (OrderCount)", min_value=0, max_value=100, value=5)
    cashback_amount = st.number_input("Rata-rata Cashback (Mata Uang)", min_value=0.0, max_value=500.0, value=150.0)

# Menghitung Fitur Turunan (Feature Engineering otomatis)
total_orders_engagement = order_count + 1  # asumsi nilai dasar kupon jika tidak diisi

# Tombol Prediksi
if st.button("🚀 Prediksi Segmen Pelanggan", type="primary"):
    # 1. Susun data input ke dalam DataFrame sesuai urutan fitur saat scaling
    input_data = pd.DataFrame([[
        tenure, 
        warehouse_to_home, 
        hour_spend, 
        order_count, 
        cashback_amount, 
        total_orders_engagement
    ]], columns=['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderCount', 'CashbackAmount', 'Total_Orders_Engagement'])

    # 2. Lakukan Scaling
    input_scaled = scaler.transform(input_data)

    # 3. Prediksi Klaster
    cluster = model.predict(input_scaled)[0]

    # 4. Tampilkan Hasil
    st.markdown("---")
    st.subheader("🎯 Hasil Prediksi Segmen")

    # Mapping Keterangan Klaster (Sesuaikan dengan profil rerata kelompokmu)
    if cluster == 0:
        st.success("### 🥇 Segmen: Pelanggan Setia / High-Value (Klaster 0)")
        st.write("**Karakteristik:** Masa berlangganan lama, transaksi tinggi, penerima cashback besar.")
        st.info("💡 **Rekomendasi Pemasaran:** Berikan akses eksklusif fitur VIP, insentif loyalty, dan penawaran preorder.")
    elif cluster == 1:
        st.warning("### 🥈 Segmen: Pelanggan Potensial / Reguler (Klaster 1)")
        st.write("**Karakteristik:** Aktivitas transaksi sedang, merespons promo dengan baik.")
        st.info("💡 **Rekomendasi Pemasaran:** Berikan voucer *cross-selling* dan promosi terbatas untuk menaikkan nilai pesanan.")
    else:
        st.error("### 🥉 Segmen: Pelanggan Baru / Berisiko Churn (Klaster 2)")
        st.write("**Karakteristik:** Tenure pendek, jarang mengakses aplikasi, nominal cashback rendah.")
        st.info("💡 **Rekomendasi Pemasaran:** Jalankan kampanye *re-engagement*, diskon onboarding, dan survei kepuasan.")