import streamlit as st

st.set_page_config(
    page_title="Prediksi Jumlah Pasien Rawat Inap",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Prediksi Jumlah Pasien Rawat Inap")

st.markdown("""
Selamat datang pada aplikasi prediksi jumlah pasien rawat inap
menggunakan tiga model machine learning:

- Prophet
- XGBoost
- Gated Recurrent Unit (GRU)

### Cara menggunakan aplikasi

1. Buka menu **Dashboard**.
2. Upload file CSV data pasien.
3. Lihat hasil analisis data historis.
4. Pilih model prediksi pada sidebar.
5. Tentukan jumlah hari prediksi.
6. Tekan tombol **Prediksi**.

---
""")

st.info(
    "Silakan pilih halaman pada sidebar untuk memulai."
)
