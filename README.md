# 🛍️ E-Commerce Customer Segmentation using Unsupervised Learning

Repository resmi proyek praktikum **Data Science** Semester Gasal 2026/2027, Program Studi Informatika, Fakultas Teknik Industri, UPN "Veteran" Yogyakarta.

---

## 👥 Anggota Kelompok

1. **Putri Asmiranti Samiun** (`123240033`)
2. **Briliant Priscilla** (`123240068`)
3. **Andini Papa Sulima** (`123240118`)
4. **Nathania Keyne Rahmawati** (`123240201`)

---

## 📋 Ringkasan Proyek

Proyek ini dirancang untuk mengelompokkan (*clustering*) pelanggan *e-commerce* berdasarkan karakteristik demografi, riwayat transaksi historis, dan pola aktivitas digital menggunakan metodologi standar **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*).

Dengan memanfaatkan algoritma *Unsupervised Learning* (**K-Means Clustering** dan **Agglomerative Hierarchical Clustering**), hasil segmentasi ini bertujuan untuk membantu tim *Marketing* dan *Customer Relationship Management* (CRM) dalam merumuskan strategi pemasaran yang terpersonalisasi, meningkatkan retensi pelanggan, serta menekan angka *churn*.

---

## 🎯 Metodologi CRISP-DM

1. **Business Understanding:** Menentukan sasaran bisnis, skenario *use-case* CRM, dan metrik evaluasi segmen.
2. **Data Understanding:** Eksplorasi struktur data (EDA), analisis sebaran variabel, visualisasi *outlier*, dan korelasi antar-fitur.
3. **Data Preparation:** Imputasi *missing values* dengan median, pembuatan fitur baru (*feature engineering*), *encoding* variabel kategorikal, dan *standardization* menggunakan `StandardScaler`.
4. **Modeling:** Penentuan jumlah klaster optimal ($k=3$) berbasis *Elbow Method* & *Silhouette Score*, dilanjutkan dengan pelatihan model K-Means dan Hierarchical Clustering (Ward linkage).
5. **Evaluation & Profiling:** Analisis performa antar-model, pemetaan karakteristik tiap segmen (*cluster profiling*), serta formulasi rekomendasi aksi bisnis.
6. **Deployment Plan:** Penyimpanan artefak model (`.pkl`), perancangan antarmuka interaktif berbasis **Streamlit**, skema monitoring *data drift*, dan strategi pelatihan ulang (*retraining*).

---

## 📁 Struktur Repositori

```text
.
├── E_Commerce_Dataset.csv          # Dataset utama E-Commerce
├── ecommerce_data_understanding.ipynb # Notebook utama (EDA hingga Evaluasi Model)
├── app.py                          # Script antarmuka aplikasi Streamlit
├── model_kmeans.pkl                # Model K-Means final (Serialized)
├── scaler.pkl                      # Scaler StandardScaler final (Serialized)
├── requirements.txt                # Daftar dependensi library Python
└── README.md                       # Dokumentasi utama repositori
```

---

## 💻 Cara Menjalankan Proyek Secara Lokal

### 1. Clone Repositori
```bash
git clone https://github.com/username-kamu/repository-kamu.git
cd repository-kamu
```

### 2. Buat & Aktifkan Virtual Environment (Opsional tetapi Direkomendasikan)
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Instal Dependensi
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi Web Streamlit
```bash
streamlit run app.py
```
Aplikasi akan secara otomatis terbuka di peramban web pada alamat `http://localhost:8501`.

---

## 🚀 Skema Deployment & Antarmuka

Aplikasi Streamlit dirancang untuk mempermudah tim operasional memasukkan data profil pelanggan baru secara *real-time* dengan skema I/O sebagai berikut:

* **Input:** Fitur `Tenure`, `WarehouseToHome`, `HourSpendOnApp`, `OrderCount`, `CashbackAmount`, serta fitur enginerring `Total_Orders_Engagement`.
* **Output:** Kategori Klaster (0, 1, atau 2), deskripsi karakteristik segmen, dan strategi pemasaran yang direkomendasikan.

### Rencana Pemantauan & Retraining:
* **Monitoring & Drift:** Evaluasi proporsi sebaran anggota klaster secara bulanan. Jika terjadi pergeseran tren atau pola transaksi (*data drift*), dilakukan pengujian statistik (PSI/KS-Test).
* **Retraining Strategy:** Pelatihan ulang model dilakukan berkala setiap **3 bulan (kuartalan)** menggunakan akumulasi data transaksi terbaru.

---

## 📜 Lisensi & Pengakuan

Proyek ini disusun sepenuhnya untuk memenuhi tugas mata kuliah **Praktikum Data Science**, Program Studi Informatika, UPN "Veteran" Yogyakarta.
