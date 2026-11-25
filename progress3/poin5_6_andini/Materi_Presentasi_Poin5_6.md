# Materi Progress 3 Poin 5 dan 6

Kontribusi Andini Papa Sulima. Bahan ini sesuai eksekusi notebook Progress3_Poin5_6_Andini.ipynb. Angka memakai data sumber commit 7cfd844e72b78874c6e232025b714bbab11b2040.

## Slide 1 Pengaturan parameter

**Tiga klaster dipilih untuk profil CRM yang lebih rinci dengan kompromi skor pemisahan**

- K-Means: uji k=2 sampai 10 dengan random_state=42 dan n_init=10.
- Hierarchical: uji single, complete, average, dan Ward pada k=2 sampai 5.
- Percobaan memakai 4.504 pelanggan latih dengan enam fitur yang sudah diskalakan.
- Pertimbangan: Silhouette, penurunan inertia, serta ukuran dan profil klaster.

| Pengaturan K-Means | Silhouette latih | Inertia | Ukuran klaster latih |
|---|---:|---:|---|
| k=2 | 0,3752 | 20.075,6 | 3.779 / 725 |
| k=3 | 0,2821 | 16.412,7 | 2.862 / 643 / 999 |

**Alasan k=3:** profil aktivitas rendah dan masa keanggotaan panjang menjadi lebih terpisah secara deskriptif. Ini pilihan untuk kebutuhan CRM, bukan skor Silhouette tertinggi.

Grafik: bukti/01_parameter_kmeans.png. Dendrogram Ward tersedia sebagai bukti cadangan pada bukti/02_dendrogram_ward.png.

### Catatan bicara sekitar 40 detik

Untuk menentukan parameter, kami menguji jumlah klaster K-Means dari dua sampai sepuluh. Kami juga mencoba empat metode linkage pada Hierarchical. Penilaiannya memakai Silhouette, inertia, dan ukuran kelompok. Skor K-Means tertinggi ada pada dua klaster, yaitu 0,3752. Namun, kami mempertahankan tiga klaster karena profil pelanggan beraktivitas rendah dan pelanggan dengan masa keanggotaan panjang menjadi lebih terpisah. Jadi, pemilihan tiga klaster merupakan pertimbangan kebutuhan CRM dengan konsekuensi skor pemisahan yang lebih rendah.

## Slide 2 Hasil modelling

**Tiga profil pelanggan terbentuk dengan pemisahan yang masih terbatas**

| Profil K-Means | Pelanggan seluruh data | Tenure rata-rata | Pesanan rata-rata |
|---|---:|---:|---:|
| Aktivitas belanja rendah | 3.599 | 6,1 bulan | 1,8 |
| Frekuensi pesanan tinggi | 792 | 11,9 bulan | 9,1 |
| Masa keanggotaan panjang | 1.239 | 20,8 bulan | 2,3 |

| Model dan subset | Silhouette |
|---|---:|
| K-Means data latih | 0,2821 |
| K-Means data uji | 0,2799 |
| Ward data latih | 0,3018 |

K-Means dipakai sebagai kandidat model aplikasi karena menyediakan predict untuk pelanggan baru. Skor uji dekat dengan skor latih pada split ini. Ward sedikit lebih tinggi pada data latih.

**Batasan:** target lama Silhouette >0,50 belum tercapai pada model final. Perlu evaluasi lanjutan dan pengujian rekomendasi pemasaran.

Grafik: bukti/03_hasil_klaster.png atau bukti/04_profil_fitur.png. Jumlah pelanggan mencakup 4.504 pelanggan latih dan 1.126 pelanggan uji.

### Catatan bicara sekitar 35 detik

Hasil K-Means membentuk tiga profil, yaitu aktivitas belanja rendah, frekuensi pesanan tinggi, dan masa keanggotaan panjang. Kelompok terbesar berisi 3.599 pelanggan. Silhouette K-Means sekitar 0,282 pada data latih dan 0,280 pada data uji. Ward sedikit lebih tinggi, tetapi K-Means mendukung penentuan segmen pelanggan baru. Pemisahan kelompoknya masih terbatas, sehingga target lama di atas 0,50 belum tercapai dan rekomendasi pemasaran masih perlu diuji.

## Bukti tambahan untuk tanya jawab

**Kenapa tidak memilih single linkage yang skornya tinggi?**
Pada k=3, skornya 0,5888 tetapi ukuran kelompok 4.501 / 2 / 1. Pada k=2, skornya 0,7664 tetapi ukuran 4.502 / 2. Sebagian besar pelanggan tetap dalam satu kelompok sehingga hasil tersebut kurang membantu membentuk segmen pemasaran. Complete dan average pada k=2 juga membentuk 4.502 / 2. Skor tinggi tetap perlu ditafsirkan bersama ukuran anggota.

**Kenapa Ward tiga klaster jika dua klaster lebih tinggi?**
Ward k=2 memperoleh 0,3397, sementara k=3 memperoleh 0,3018. Tiga klaster menjadi pembanding pada jumlah segmen yang sama dengan K-Means. Jadi, jangan menyebut Ward k=3 sebagai maksimum Silhouette.

**Apa bukti tiga klaster memberi profil tambahan?**
Pada k=2, kelompok mayoritas memiliki Tenure rata-rata 9,62 bulan dan pesanan 1,89. Pada k=3, muncul kelompok Tenure 6,04 bulan dan kelompok Tenure 20,85 bulan pada data latih. Tabel lengkap dan tabulasi silang ada di profil_k2_vs_k3.csv dan keanggotaan_k2_vs_k3.csv. K-Means dilatih ulang sehingga pembagiannya bukan pecahan bersarang yang persis.

**Apa fungsi random_state dan n_init?**
random_state=42 untuk reproduksibilitas. n_init=10 berarti sepuluh percobaan inisialisasi, lalu dipilih inertia terendah. Angka 42 tidak dipilih karena dianggap seed dengan skor terbaik.

**Apakah Silhouette 0,282 berarti akurasi 28,2 persen?**
Tidak. Silhouette berada pada rentang -1 sampai 1 dan menilai pemisahan serta kekompakan klaster, tanpa label segmen acuan.

**Apakah data uji dipakai memilih parameter?**
Tidak pada notebook lanjutan ini. Pilihan parameter memakai data latih; data uji digunakan setelah k=3 ditetapkan. Namun, split yang sama sudah pernah diperiksa di proyek kelompok, sehingga belum menjadi holdout baru yang sepenuhnya independen.

**Apa yang diperbaiki pada preprocessing?**
Split dilakukan sebelum imputasi. Median dan scaler dipelajari dari data latih saja. Semua median latih sama dengan median seluruh data pada pendekatan lama, sehingga hasil numerik tetap konsisten dengan proyek.

**Apakah churn dipakai saat training?**
Tidak. Churn hanya ditampilkan pada profil historis setelah klaster terbentuk. Perbedaan churn antarklaster tidak membuktikan penyebab churn atau memprediksi risiko setiap individu.

**Mengapa tidak menyebut cashback tinggi sebagai profit tinggi?**
Dataset menampilkan cashback, bukan laba atau Customer Lifetime Value. Penamaan profil dibuat deskriptif agar sesuai bukti.

**Apa batas kontribusi saya?**
Saya mengerjakan percobaan parameter, dasar pemilihan pengaturan, dan hasil awal berupa metrik, ukuran anggota, profil serta grafik. Tabel perbandingan dan pembahasan akhir dapat dilanjutkan oleh penanggung jawab poin 7 dan 8.
