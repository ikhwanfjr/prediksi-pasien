import streamlit as st
import pandas as pd

from utils.file_handler import load_historical_data

from utils.analysis import analyze_patient_trends

from utils.visualization import (
    plot_history,
    plot_average_day,
    plot_average_month,
    plot_total_year,
)

st.set_page_config(
    page_title="Dashboard",
    layout="wide",
)

st.title(
    "Dashboard Prediksi Pasien Rawat Inap"
)

st.write(
    """
    Upload data rawat inap untuk melihat
    analisis historis dan melakukan prediksi.
    """
)

uploaded_file = st.file_uploader(
    "Upload Data Rawat Inap (.csv)",
    type="csv",
)

if uploaded_file is None:

    st.info(
        "Silakan upload dataset terlebih dahulu."
    )

    st.stop()

try:

    historical_df = load_historical_data(
        uploaded_file
    )

    historical_df["tanggal"] = pd.to_datetime(
        historical_df["tanggal"]
    )

except Exception as e:

    st.error(str(e))

    st.stop()

st.session_state["historical_df"] = historical_df

summary = analyze_patient_trends(
    historical_df
)



col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Jumlah Hari",
        len(historical_df)
    )

with col2:
    st.metric(
        "Total Pasien",
        int(historical_df["jumlah_pasien"].sum())
    )

with col3:
    st.metric(
        "Tanggal Awal",
        historical_df["tanggal"].min().strftime("%d-%m-%Y")
    )

with col4:
    st.metric(
        "Tanggal Akhir",
        historical_df["tanggal"].max().strftime("%d-%m-%Y")
    )

st.subheader("Data Historis")
st.plotly_chart(

    plot_history(historical_df),

    use_container_width=True
)

st.subheader("Data Hasil Preparation")

st.dataframe(
    historical_df.head(10),
    use_container_width=True
)

st.subheader("Rata-Rata Pasien Per Hari")
st.plotly_chart(

    plot_average_day(
        summary["daily_average"]
    ),

    use_container_width=True
)

st.subheader("Rata-Rata Pasien Per Bulan")
st.plotly_chart(

    plot_average_month(
        summary["monthly_average"]
    ),

    use_container_width=True
)

st.subheader("Rata-Rata Pasien Per Tahun")
st.plotly_chart(

    plot_total_year(
        summary["yearly_total"]
    ),

    use_container_width=True
)