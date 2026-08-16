import streamlit as st

from predictors.gru_predict import predict_gru
from utils.load_resources import load_gru
from utils.visualization import plot_forecast

st.set_page_config(
    page_title="Prediksi GRU",
    layout="wide"
)

st.title("Prediksi Jumlah Pasien Menggunakan GRU")

st.write("""
Halaman ini digunakan untuk melakukan prediksi jumlah pasien
menggunakan model GRU yang telah dilatih.
""")

# ======================================================
# Cek Data
# ======================================================

if "historical_df" not in st.session_state:

    st.warning(
        "Silakan upload data pada halaman Dashboard."
    )

    st.stop()

historical_df = st.session_state["historical_df"]

st.subheader("Data Historis")

st.dataframe(
    historical_df.head(10),
    use_container_width=True
)

# ======================================================
# Load Model
# ======================================================

@st.cache_resource
def load_model():

    return load_gru()

gru_model, scaler_X, scaler_y, feature_list = load_model()

# ======================================================
# Input
# ======================================================

st.subheader("Parameter Prediksi")

periods = st.number_input(
    "Jumlah Hari Prediksi",
    min_value=1,
    max_value=365,
    value=30,
    step=1
)

# ======================================================
# Prediksi
# ======================================================

if st.button("Prediksi", type="primary"):

    with st.spinner("Melakukan prediksi..."):

        prediction_df = predict_gru(
            model=gru_model,
            historical_df=historical_df,
            periods=periods,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            feature_list=feature_list
        )

    # ==================================================
    # Ringkasan
    # ==================================================

    st.subheader("Ringkasan Prediksi")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Prediksi",
            int(prediction_df["prediksi"].sum())
        )

    with col2:

        st.metric(
            "Rata-rata",
            f"{prediction_df['prediksi'].mean():.2f}"
        )

    with col3:

        st.metric(
            "Maksimum",
            int(prediction_df["prediksi"].max())
        )

    with col4:

        st.metric(
            "Minimum",
            int(prediction_df["prediksi"].min())
        )

    # ==================================================
    # Grafik
    # ==================================================

    st.subheader("Grafik Prediksi")

    fig = plot_forecast(
        historical_df,
        prediction_df
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==================================================
    # Tabel
    # ==================================================

    st.subheader("Hasil Prediksi")

    st.dataframe(
        prediction_df,
        use_container_width=True
    )

    # ==================================================
    # Download
    # ==================================================

    csv = prediction_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📥 Download Hasil Prediksi",
        data=csv,
        file_name="hasil_prediksi_gru.csv",
        mime="text/csv"
    )