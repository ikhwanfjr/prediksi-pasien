# Prediksi Jumlah Pasien Rawat Inap RSUD Pringsewu

## 1. Tentang Projek Penelitian

Proyek ini merupakan implementasi penelitian untuk memprediksi jumlah pasien rawat inap di RSUD Pringsewu menggunakan pendekatan *time series forecasting*. Penelitian ini bertujuan untuk membandingkan performa tiga algoritma, yaitu:

- **Prophet** – Traditional Time Series Forecasting
- **XGBoost** – Machine Learning
- **GRU (Gated Recurrent Unit)** – Deep Learning

Penelitian menggunakan data historis pasien rawat inap periode Januari 2023 hingga Maret 2026. Data awal terdiri dari 29.459 baris yang kemudian melalui tahap *data preparation* sehingga menghasilkan 1.156 baris data dengan 14 fitur.

Metodologi penelitian menggunakan **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*) yang terdiri dari enam tahapan:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

Model dengan performa terbaik dipilih untuk diimplementasikan ke dalam aplikasi berbasis Streamlit sebagai sarana visualisasi dan prediksi jumlah pasien rawat inap.

## 2. Struktur Repository

Repository ini mencakup file pemodelan, hasil pelatihan model, dataset, serta implementasi aplikasi Streamlit.

```text
prediksi-pasien/
│
├── preprocessing/
│   ├── aggregation.py
│   ├── features_engineering.py
│   └── lag_features.py
│
├── modeling/
│   ├── prophet/
│   ├── xgboost/
│   └── gru/
│
├── models/
│   ├── xgboost_model.json
│   ├── features.json
│   └── ...
│
├── dataset/
│   └── ...
│
├── future_dates.py
├── xgboost_predict.py
├── gru_predict.py
├── gru_sequence.py
├── gru_features.py
├── load_resources.py
├── file_handler.py
├── analysis.py
├── visualization.py
├── app.py
├── requirements.txt
└── README.md
```

## 3. Pipeline Utama

Proses penelitian dan pengembangan model dilakukan melalui beberapa tahapan berikut:

### 3.1 Data Preparation

- Memilih atribut yang diperlukan dari data pasien rawat inap.
- Mengolah tanggal masuk dan tanggal keluar pasien untuk membentuk data jumlah pasien harian.
- Melakukan agregasi jumlah pasien berdasarkan tanggal.
- Melakukan pemeriksaan dan penanganan kualitas data.
- Membentuk fitur kalender dan hari libur nasional.
- Membentuk fitur historis (*lag features*) untuk kebutuhan pemodelan.

### 3.2 Modeling

Membangun tiga model prediksi menggunakan algoritma Prophet, XGBoost, dan GRU.

Optimasi *hyperparameter* dilakukan menggunakan **Optuna** dengan validasi **Time Series Split** untuk memperoleh kombinasi parameter terbaik pada masing-masing model.

Pembagian data menggunakan komposisi:

- Data latih: 70% (809 baris).
- Data uji: 30% (347 baris).

### 3.3 Evaluation

Evaluasi dilakukan dengan membandingkan performa ketiga model menggunakan empat metrik:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Weighted Absolute Percentage Error (WAPE)

Model dengan nilai kesalahan paling rendah dipilih sebagai model terbaik.

### 3.4 Deployment

Model terbaik diimplementasikan ke dalam aplikasi berbasis Streamlit untuk melakukan prediksi jumlah pasien rawat inap dan menyajikan hasilnya dalam bentuk visualisasi.

## 4. Hasil Skor Evaluasi

Berdasarkan hasil pengujian pada data uji, diperoleh perbandingan performa sebagai berikut:

| Metrik | GRU | XGBoost | Prophet |
|:---|---:|---:|---:|
| MAE | 10,4 | **8** | 15,8 |
| RMSE | 13,6 | **10,2** | 20,4 |
| MAPE | 11,2% | **8,4%** | 17,8% |
| WAPE | 10,1% | **7,7%** | 15,1% |

Berdasarkan hasil evaluasi, **XGBoost memperoleh performa terbaik** dibandingkan Prophet dan GRU karena menghasilkan nilai kesalahan paling rendah pada seluruh metrik evaluasi.

Oleh karena itu, XGBoost dipilih sebagai model yang diimplementasikan pada aplikasi prediksi jumlah pasien rawat inap.

## 5. Dependensi Utama

Proyek ini dikembangkan menggunakan Python dengan beberapa library utama berikut:

| Library | Kegunaan |
|:---|:---|
| Pandas | Pengolahan dan manipulasi data |
| NumPy | Operasi numerik |
| Scikit-learn | Preprocessing dan evaluasi model |
| XGBoost | Pemodelan machine learning |
| TensorFlow / Keras | Pemodelan deep learning menggunakan GRU |
| Prophet | Pemodelan time series forecasting |
| Optuna | Optimasi hyperparameter |
| Holidays | Identifikasi hari libur nasional |
| Streamlit | Pengembangan aplikasi web |
| Matplotlib | Visualisasi data dan hasil prediksi |

Untuk memasang seluruh dependensi, jalankan perintah berikut:

```bash
pip install -r requirements.txt
```

## 6. Fitur Web Streamlit

Aplikasi prediksi jumlah pasien rawat inap dikembangkan menggunakan Streamlit dengan fitur utama sebagai berikut:

- **Upload Dataset**: Mengunggah data pasien rawat inap dalam format CSV.
- **Data Processing**: Melakukan pengolahan data berdasarkan tanggal masuk dan tanggal keluar pasien.
- **Historical Dashboard**: Menampilkan ringkasan dan visualisasi historis jumlah pasien rawat inap.
- **Model Prediction**: Menghasilkan prediksi jumlah pasien rawat inap menggunakan model yang tersedia.
- **Forecast Visualization**: Menampilkan hasil prediksi dalam bentuk grafik.
- **Data Analysis**: Menyajikan informasi statistik dari data historis pasien rawat inap.

### Menjalankan Aplikasi

Pastikan seluruh dependensi telah terpasang, kemudian jalankan perintah:

```bash
streamlit run app.py
```

Aplikasi akan berjalan secara lokal dan dapat diakses melalui browser.

---

## Catatan

Proyek ini dikembangkan untuk keperluan penelitian akademik dan sebagai prototipe sistem prediksi jumlah pasien rawat inap. Hasil prediksi ditujukan sebagai informasi pendukung dalam perencanaan kebutuhan bahan makanan di Instalasi Gizi RSUD Pringsewu, bukan sebagai pengganti keputusan operasional rumah sakit.

Dataset yang mengandung informasi pribadi pasien tidak boleh disebarluaskan secara publik tanpa izin dan prosedur perlindungan data yang sesuai.
