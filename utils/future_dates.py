import pandas as pd


def create_future_dates(last_historical_date, periods):
    """
    Membuat DataFrame berisi tanggal prediksi.

    Parameters
    ----------
    last_date : str | datetime
        Tanggal terakhir pada data historis.
    periods : int
        Jumlah hari yang akan diprediksi.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan satu kolom:
        - tanggal
    """

    last_historical_date = pd.to_datetime(last_historical_date)

    future_dates = pd.date_range(
        start=last_historical_date + pd.Timedelta(days=1),
        periods=periods,
        freq="D"
    )

    future_df = pd.DataFrame({
        "tanggal": future_dates
    })

    return future_df