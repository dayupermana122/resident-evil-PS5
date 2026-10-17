Prediksi Penjualan Game Resident Evil (PS5)
Deskripsi Proyek
Proyek ini mengimplementasikan alur kerja machine learning menggunakan bahasa Python dan pustaka Scikit-Learn untuk memprediksi total penjualan (dalam jutaan kopi) dari judul-judul game Resident Evil di platform PS5. Prediksi dilakukan berdasarkan berbagai fitur seperti skor Metacritic, skor pengguna, harga rilis, durasi permainan, serta format dan perspektif game. Proyek ini menggunakan dataset resident_evil_ps5.csv.
Struktur Direktori
main.py : Skrip utama yang berisi seluruh kode machine learning.
resident_evil_ps5.csv : Dataset yang digunakan untuk melatih dan menguji model.
output/ : Folder yang dibuat secara otomatis oleh skrip untuk menyimpan hasil grafik (EDA), tabel evaluasi, dan model akhir.
Alur Kerja (Workflow)
Skrip main.py akan menjalankan tahapan berikut secara berurutan:

Data Loading: Membaca dataset dan menampilkan informasi statistik deskriptif awal.
Data Cleaning: Menghapus data duplikat dan baris data yang nilai targetnya (Total_Copies_Sold_Millions) kosong.
Exploratory Data Analysis (EDA): Membuat visualisasi distribusi data (Histogram), mendeteksi outlier (Boxplot), dan menganalisis korelasi antar fitur numerik (Heatmap). Hasil grafik disimpan di folder output/.
Data Splitting: Membagi dataset menjadi data latih (80%) dan data uji (20%).
Preprocessing & Modelling:
Memproses nilai yang hilang (missing values) menggunakan median untuk fitur numerik dan modus untuk fitur kategorikal.
Melakukan standardisasi skala pada fitur numerik.
Menerapkan One-Hot Encoding pada fitur kategorikal.
Melatih tiga model regresi untuk dibandingkan performanya.
Evaluasi dan Penyimpanan: Menghitung metrik performa (MAE, MSE, RMSE, $R^2$), lalu menyimpan ringkasannya serta model terbaik yang didapatkan.
Model yang Digunakan
Proyek ini membandingkan kinerja tiga algoritma regresi:

Least Angle Regression (LARS)
Linear Regression
Gradient Boosting Regressor
Hasil Output
Setelah skrip selesai dieksekusi, periksa folder output/ untuk melihat file-file berikut:

01_histogram.png : Distribusi fitur numerik dan target.
02_boxplot.png : Deteksi nilai ekstrem (outlier).
03_heatmap_korelasi.png : Visualisasi hubungan korelasi antar fitur.
hasil_evaluasi_model.csv : Tabel perbandingan metrik kesalahan dari ketiga model.
perbandingan_prediksi.csv : Tabel perbandingan antara nilai target aktual dengan hasil prediksi model terbaik pada data uji.
model_terbaik.joblib : File model terlatih dengan nilai Error terkecil, yang dapat dimuat ulang (load) untuk memprediksi data baru.
Cara Menjalankan
Pastikan Anda telah menginstal pustaka yang dibutuhkan (pandas, numpy, matplotlib, seaborn, scikit-learn, joblib). Buka terminal di direktori proyek ini, lalu jalankan perintah:

python main.py
