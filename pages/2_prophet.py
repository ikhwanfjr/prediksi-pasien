import streamlit as st

from predictors.prophet_predict import predict_prophet
from utils.load_resources import load_prophet
from utils.visualization import plot_forecast

st.set_page_config(
    page_title="Prediksi Prophet",
    layout="wide"
)

st.title("Prediksi Jumlah Pasien Menggunakan Prophet")

# ==========================================================
# Cek data historis
# ==========================================================

if "historical_df" not in st.session_state:

    st.warning(
        "Silakan upload data terlebih dahulu melalui halaman Dashboard."
    )

    st.stop()

historical_df = st.session_state["historical_df"]

st.subheader("Data Historis")

st.dataframe(
    historical_df.head(10),
    use_container_width=True
)

# ==========================================================
# Load Model
# ==========================================================

@st.cache_resource
def load_model():
    return load_prophet()

prophet_model, prophet_regressors = load_model()

# ==========================================================
# Input
# ==========================================================

st.subheader("Parameter Prediksi")

periods = st.number_input(
    "Jumlah Hari Prediksi",
    min_value=1,
    max_value=365,
    value=30,
    step=1
)

# ==========================================================
# Prediksi
# ==========================================================

if st.button("Prediksi", type="primary"):

    with st.spinner("Melakukan prediksi..."):

        prediction_df = predict_prophet(
            model=prophet_model,
            historical_df=historical_df,
            periods=periods,
            prophet_regressors=prophet_regressors
        )

    # ======================================================
    # Ringkasan
    # ======================================================

    st.subheader("Ringkasan Prediksi")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Prediksi",
            f"{prediction_df['prediksi'].sum():.0f}"
        )

    with col2:
        st.metric(
            "Rata-rata",
            f"{prediction_df['prediksi'].mean():.2f}"
        )

    with col3:
        st.metric(
            "Maksimum",
            f"{prediction_df['prediksi'].max():.0f}"
        )

    with col4:
        st.metric(
            "Minimum",
            f"{prediction_df['prediksi'].min():.0f}"
        )

    # ======================================================
    # Grafik
    # ======================================================

    st.subheader("Grafik Prediksi")

    fig = plot_forecast(
        historical_df,
        prediction_df
    )
    
    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ======================================================
    # Tabel
    # ======================================================

    st.subheader("Hasil Prediksi")

    st.dataframe(
        prediction_df,
        use_container_width=True
    )

    # ======================================================
    # Download
    # ======================================================

    csv = prediction_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Hasil Prediksi",
        data=csv,
        file_name="hasil_prediksi_prophet.csv",
        mime="text/csv"
    )