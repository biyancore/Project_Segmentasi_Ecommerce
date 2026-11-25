# Progress 3 Poin 5 dan 6

Notebook lanjutan untuk Andini Papa Sulima, proyek Segmentasi Pelanggan E Commerce.

## Isi paket

- Progress3_Poin5_6_Andini.ipynb: notebook mandiri yang sudah dijalankan dengan tabel dan grafik tersimpan.
- E_Commerce_Dataset.csv: dataset dari commit sumber, disertakan agar Run All tidak memerlukan unduhan.
- Materi_Presentasi_Poin5_6.md: isi dua slide, catatan bicara, dan persiapan tanya jawab.
- bukti: tabel CSV, empat grafik PNG, keanggotaan pelanggan, dan ringkasan versi beserta hash data.

## Menjalankan ulang

Ekstrak seluruh ZIP dalam satu folder. Buka notebook dengan Jupyter atau VS Code, lalu pilih Restart Kernel and Run All. Pastikan direktori kerja adalah folder yang berisi notebook dan CSV. Hasil tabel dan gambar pada folder bukti akan diperbarui. Tidak ada file model aplikasi yang ditulis.

Lingkungan eksekusi yang dipakai tercatat pada requirements_analisis.txt serta bukti/ringkasan.json. Jika membuat lingkungan Python baru, instal dependensi dari requirements_analisis.txt.

## Lingkup hasil

Poin 5: K-Means k=2 sampai 10; empat linkage Hierarchical pada k=2 sampai 5; Silhouette, inertia, ukuran kelompok, dendrogram; perbandingan profil k=2 dan k=3; alasan serta kompromi pilihan tiga klaster.

Poin 6: metrik model final, keanggotaan latih dan uji, profil K-Means seluruh pelanggan, profil Ward pada latih, grafik sebar dan heatmap enam fitur, serta keterkaitan dengan kebutuhan CRM.

Median dan scaler dihitung dari data latih saja. Median ternyata identik dengan pendekatan sebelumnya. Hasil K-Means k=3 adalah 0,2821 pada latih dan 0,2799 pada uji; Ward 0,3018 pada latih. Target lama >0,50 belum tercapai pada model final. k=3 merupakan pilihan kebutuhan profil bisnis, bukan skor terbaik.

Sumber repository: https://github.com/biyancore/Project_Segmentasi_Ecommerce

Commit sumber: 7cfd844e72b78874c6e232025b714bbab11b2040

Sumber dataset yang dicantumkan proyek: Ankit Verma, Ecommerce Customer Churn Analysis and Prediction, Kaggle, https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction

Dataset mengikuti keterangan lisensi proyek CC BY-NC-SA 4.0. Analisis ini untuk tugas akademik nonkomersial dengan atribusi sumber.

## Integrasi ke proyek kelompok

Tambahkan notebook lanjutan dan dataset yang sama ke folder proyek atau salin hanya notebook dan materi jika dataset sudah tersedia. Notebook sumber, aplikasi, dan artefak model tidak diganti. Direktori ini disiapkan untuk ditambahkan melalui pull request ke repository kelompok. Tidak ada PowerPoint atau Canva yang diubah; materi presentasi berupa teks siap salin.
