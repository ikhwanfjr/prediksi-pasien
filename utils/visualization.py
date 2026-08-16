import plotly.graph_objects as go
import plotly.express as px


def plot_history(df):
    """
    Grafik data historis pasien.
    """

    fig = px.line(
        df,
        x="tanggal",
        y="jumlah_pasien",
        title="Data Historis Jumlah Pasien",
        markers=True,
    )

    fig.update_layout(
        xaxis_title="Tanggal",
        yaxis_title="Jumlah Pasien",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


def plot_forecast(history_df, forecast_df):
    """
    Grafik historis + forecast.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history_df["tanggal"],
            y=history_df["jumlah_pasien"],
            mode="lines",
            name="Historis",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["tanggal"],
            y=forecast_df["prediksi"],
            mode="lines+markers",
            name="Forecast",
        )
    )

    fig.update_layout(
        title="Prediksi Jumlah Pasien",
        xaxis_title="Tanggal",
        yaxis_title="Jumlah Pasien",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


def plot_average_day(df):
    """
    Bar chart rata-rata pasien per hari.
    """

    fig = px.bar(
        df,
        x="hari",
        y="rata_rata",
        title="Rata-rata Pasien per Hari",
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Hari",
        yaxis_title="Rata-rata Pasien",
    )

    return fig


def plot_average_month(df):
    """
    Bar chart rata-rata pasien per bulan.
    """

    fig = px.bar(
        df,
        x="bulan",
        y="rata_rata",
        title="Rata-rata Pasien per Bulan",
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Bulan",
        yaxis_title="Rata-rata Pasien",
    )

    return fig


def plot_total_year(df):
    """
    Total pasien tiap tahun.
    """

    fig = px.line(
        df,
        x="tahun",
        y="total_pasien",
        markers=True,
        title="Total Pasien per Tahun",
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Tahun",
        yaxis_title="Total Pasien",
    )

    return fig


def prediction_summary(forecast_df):
    """
    Ringkasan hasil prediksi.
    """

    return {
        "total": int(forecast_df["prediksi"].sum()),
        "average": round(forecast_df["prediksi"].mean(), 2),
        "maximum": int(forecast_df["prediksi"].max()),
        "minimum": int(forecast_df["prediksi"].min()),
    }