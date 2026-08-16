import pandas as pd
import numpy as np

from preprocessing.features_engineering import feature_engineering
from preprocessing.lag_features import (
    build_lag_features,
    update_target_history,
)

from preprocessing.gru_sequence import (
    create_sequence,
)

from utils.future_dates import create_future_dates

from preprocessing.gru_features import prepare_gru_features

LAG_COLUMNS= [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_30",
]


def predict_gru(
    model,
    historical_df,
    scaler_X,
    scaler_y,
    feature_list,
    periods=30,
    time_steps=30,
):
    """
    Melakukan recursive forecasting menggunakan model GRU.
    """

    # =====================================================
    # Copy dataframe
    # =====================================================

    historical_df = historical_df.copy()

    # =====================================================
    # Validasi kolom
    # =====================================================

    required_columns = {
        "tanggal",
        "jumlah_pasien",
    }

    missing_columns = (
        required_columns -
        set(historical_df.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Kolom berikut tidak ditemukan: {missing_columns}"
        )

    # =====================================================
    # Pastikan datetime
    # =====================================================

    historical_df["tanggal"] = pd.to_datetime(
        historical_df["tanggal"]
    )

    # =====================================================
    # Future dates
    # =====================================================

    future_df = create_future_dates(
        last_historical_date=historical_df["tanggal"].max(),
        periods=periods,
    )

    # =====================================================
    # Feature engineering future dates
    # =====================================================

    future_df = feature_engineering(
        future_df
    )

    # =====================================================
    # History target
    # =====================================================

    history = historical_df[
        "jumlah_pasien"
    ].tolist()

    predictions = []

    # =====================================================
    # Menyiapkan feature matrix historis
    # =====================================================

    feature_matrix = prepare_gru_features(
        historical_df=historical_df,
        scaler_X=scaler_X,
        feature_list=feature_list,
    )

    # =====================================================
    # Membentuk sequence awal
    # =====================================================

    sequence = create_sequence(
        feature_matrix=feature_matrix,
        time_steps=time_steps,
    )

    # =====================================================
    # Recursive Forecasting
    # =====================================================

    for i in range(periods):

        # ---------------------------------------------
        # Ambil satu baris future
        # ---------------------------------------------

        feature_row = future_df.iloc[[i]].copy()

        # ---------------------------------------------
        # Membentuk fitur lag
        # ---------------------------------------------

        lag_features = build_lag_features(
            history
        )

        # ---------------------------------------------
        # Menambahkan lag ke feature row
        # ---------------------------------------------

        for column, value in lag_features.items():

            feature_row[column] = value

        # ---------------------------------------------
        # Validasi fitur
        # ---------------------------------------------

        missing_features = [
            col
            for col in feature_list
            if col not in feature_row.columns
        ]

        if missing_features:

            raise ValueError(
                f"Fitur berikut belum tersedia: {missing_features}"
            )

        # ---------------------------------------------
        # Urutkan sesuai training
        # ---------------------------------------------

        feature_row = feature_row[
            feature_list
        ]

        # ---------------------------------------------
        # Scaling hanya fitur lag
        # ---------------------------------------------

        feature_row_scaled = feature_row.copy()

        feature_row_scaled[LAG_COLUMNS] = (
            scaler_X.transform(
                feature_row_scaled[LAG_COLUMNS]
            )
        )

        # ---------------------------------------------
        # Prediksi menggunakan GRU
        # ---------------------------------------------

        pred_scaled = model.predict(
            sequence,
            verbose=0
        )

        # ---------------------------------------------
        # Inverse scaling
        # ---------------------------------------------

        prediction = scaler_y.inverse_transform(
            pred_scaled
        )[0, 0]

        predictions.append(
            prediction
        )

        # ---------------------------------------------
        # Update history target
        # ---------------------------------------------

        max_history = max(
            int(col.split("_")[1])
            for col in feature_list
            if col.startswith("lag_")
        )

        history = update_target_history(
            history=history,
            prediction=prediction,
            max_history=max_history
        )

                # ---------------------------------------------
        # Jika masih ada tanggal berikutnya,
        # tambahkan feature row baru ke feature_matrix
        # ---------------------------------------------

        if i < periods - 1:

            next_feature_row = future_df.iloc[[i + 1]].copy()

            # Build lag berdasarkan history terbaru
            new_lag_features = build_lag_features(history)

            for column, value in new_lag_features.items():
                next_feature_row[column] = value

            # Susun urutan fitur
            next_feature_row = next_feature_row[
                feature_list
            ]

            # Scaling hanya kolom lag
            next_feature_row_scaled = next_feature_row.copy()

            next_feature_row_scaled[LAG_COLUMNS] = (
                scaler_X.transform(
                    next_feature_row_scaled[LAG_COLUMNS]
                )
            )

            # Tambahkan ke feature matrix
            feature_matrix = np.vstack([
                feature_matrix,
                next_feature_row_scaled.values
            ])

            # Bentuk ulang sequence
            sequence = create_sequence(
                feature_matrix=feature_matrix,
                time_steps=time_steps
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
            "prediksi"
        ]
    ]