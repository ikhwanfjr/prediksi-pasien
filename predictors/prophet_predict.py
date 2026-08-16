import pandas as pd

from preprocessing.features_engineering import feature_engineering
from utils.future_dates import create_future_dates


def predict_prophet(
    model,
    historical_df,
    prophet_regressors,
    periods=30
):
    """
    Melakukan prediksi menggunakan model Prophet.

    Parameters
    ----------
    model : Prophet
        Model Prophet yang telah di-load.

    historical_df : pd.DataFrame
        Data historis hasil preprocessing.

        Minimal memiliki kolom:
        - tanggal
        - jumlah_pasien

    prophet_regressors : list
        Daftar nama regressor Prophet.

    periods : int, default=30
        Jumlah hari yang akan diprediksi.

    Returns
    -------
    pd.DataFrame

        Kolom:
        - ds
        - yhat
        - yhat_lower
        - yhat_upper
    """

    # =====================================================
    # Copy dataframe
    # =====================================================

    historical_df = historical_df.copy()

    # =====================================================
    # Validasi kolom
    # =====================================================

    required_columns = {"tanggal", "jumlah_pasien"}

    missing_columns = required_columns - set(historical_df.columns)

    if missing_columns:
        raise ValueError(
            f"Kolom berikut tidak ditemukan: {missing_columns}"
        )

    # =====================================================
    # Pastikan bertipe datetime
    # =====================================================

    historical_df["tanggal"] = pd.to_datetime(
        historical_df["tanggal"]
    )

    # =====================================================
    # Membuat future dates
    # =====================================================

    future_df = create_future_dates(
        last_historical_date=historical_df["tanggal"].max(),
        periods=periods
    )

    # =====================================================
    # Feature engineering
    # =====================================================

    future_df = feature_engineering(future_df)

    # =====================================================
    # Rename sesuai format Prophet
    # =====================================================

    future_df = future_df.rename(
        columns={
            "tanggal": "ds"
        }
    )

    # =====================================================
    # Validasi regressor
    # =====================================================

    missing_regressors = [
        col for col in prophet_regressors
        if col not in future_df.columns
    ]

    if missing_regressors:
        raise ValueError(
            f"Regressor berikut belum tersedia: {missing_regressors}"
        )

    # =====================================================
    # Menyusun dataframe sesuai model Prophet
    # =====================================================

    future_df = future_df[
        ["ds"] + prophet_regressors
    ]

    # =====================================================
    # Prediksi
    # =====================================================

    forecast = model.predict(future_df)

    # =====================================================
    # Mengembalikan hasil yang diperlukan
    # =====================================================

    result = pd.DataFrame(
    {
        "tanggal": forecast["ds"],
        "prediksi": forecast["yhat"]
    }
    )

    result["prediksi"] = (
    result["prediksi"]
    .clip(lower=0)
    .round()
    .astype(int)
    )

    return result