import pandas as pd

from preprocessing.features_engineering import feature_engineering
from preprocessing.lag_features import (
    update_target_history,
    build_lag_features,
)
from utils.future_dates import create_future_dates


def predict_xgboost(
    model,
    historical_df,
    feature_list,
    periods=30,
):
    """
    Melakukan recursive forecasting menggunakan XGBoost.

    Parameters
    ----------
    model : XGBRegressor

    historical_df : pd.DataFrame

        Minimal memiliki kolom:
        - tanggal
        - jumlah_pasien

    feature_list : list
        Daftar fitur sesuai model training.

    periods : int
        Horizon prediksi.

    Returns
    -------
    pd.DataFrame
    """

    historical_df = historical_df.copy()

    historical_df["tanggal"] = pd.to_datetime(
        historical_df["tanggal"]
    )

    # =====================================================
    # Membuat tanggal masa depan
    # =====================================================

    future_df = create_future_dates(
        last_historical_date=historical_df["tanggal"].max(),
        periods=periods,
    )

    # =====================================================
    # Feature kalender
    # =====================================================

    future_df = feature_engineering(future_df)

    # =====================================================
    # Riwayat target
    # =====================================================

    history = historical_df["jumlah_pasien"].tolist()

    predictions = []

    # =====================================================
    # Recursive Forecasting
    # =====================================================

    for i in range(len(future_df)):

        row = future_df.iloc[[i]].copy()

        lag_features = build_lag_features(history)

        for key, value in lag_features.items():
            row[key] = value

        row = row[feature_list]

        prediction = model.predict(row)[0]

        predictions.append(prediction)

        history = update_target_history(
            history,
            prediction,
            max_history=30,
        )

    # =====================================================
    # Hasil prediksi
    # =====================================================

    result = future_df.copy()

    result["prediksi"] = (
    pd.Series(predictions)
    .clip(lower=0)
    .round()
    .astype(int)
    )

    return result[
        [
            "tanggal",
            "prediksi",
        ]
    ]