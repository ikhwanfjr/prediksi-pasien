import pandas as pd

from preprocessing.features_engineering import feature_engineering
from preprocessing.lag_features import create_lag_features


LAG_COLUMNS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_30",
]


def prepare_gru_features(
    historical_df,
    scaler_X,
    feature_list,
):
    """
    Menyiapkan feature matrix untuk model GRU.

    Pipeline sama seperti saat training:

    1. Feature Engineering
    2. Create Lag Features
    3. Scaling hanya pada kolom lag
    4. Susun sesuai urutan feature_list

    Parameters
    ----------
    historical_df : pd.DataFrame

        Minimal memiliki kolom:
        - tanggal
        - jumlah_pasien

    scaler_X : MinMaxScaler

        Scaler yang digunakan saat training.

    feature_list : list

        Daftar fitur dari features.json

    Returns
    -------
    pd.DataFrame

        Feature matrix yang sudah siap
        digunakan untuk create_sequence().
    """

    df = historical_df.copy()

    # ==========================================
    # Validasi
    # ==========================================

    required_columns = {
        "tanggal",
        "jumlah_pasien",
    }

    missing_columns = (
        required_columns -
        set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Kolom berikut tidak ditemukan: {missing_columns}"
        )

    df["tanggal"] = pd.to_datetime(
        df["tanggal"]
    )

    # ==========================================
    # Feature Engineering
    # ==========================================

    df = feature_engineering(df)

    # ==========================================
    # Lag Features
    # ==========================================

    df = create_lag_features(
        df,
        target_col="jumlah_pasien",
        dropna=True,
    )

    # ==========================================
    # Validasi feature
    # ==========================================

    missing_features = [
        col
        for col in feature_list
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Feature berikut tidak ditemukan: {missing_features}"
        )

    # ==========================================
    # Scaling hanya kolom lag
    # ==========================================

    df_scaled = df.copy()

    df_scaled[LAG_COLUMNS] = scaler_X.transform(
        df_scaled[LAG_COLUMNS]
    )

    # ==========================================
    # Susun urutan feature
    # ==========================================

    feature_matrix = df_scaled[
        feature_list
    ].values

    return feature_matrix