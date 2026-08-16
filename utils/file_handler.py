import pandas as pd

from preprocessing.aggregation import aggregate_daily
from preprocessing.features_engineering import feature_engineering
from preprocessing.lag_features import create_lag_features


REQUIRED_COLUMNS = [
    "no_register",
    "tgl_masuk",
    "tgl_keluar",
]


def validate_columns(df: pd.DataFrame):

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Kolom berikut tidak ditemukan: "
            f"{', '.join(missing)}"
        )


def load_historical_data(uploaded_file):

    if uploaded_file is None:
        raise ValueError(
            "Silakan upload file CSV terlebih dahulu."
        )

    df = pd.read_csv(uploaded_file)

    if df.empty:
        raise ValueError(
            "File yang diupload kosong."
        )

    validate_columns(df)

    # =====================================================
    # 1. AGREGASI DATA
    # =====================================================

    daily = aggregate_daily(df)

    # =====================================================
    # 2. FEATURE ENGINEERING
    # =====================================================

    daily = feature_engineering(daily)

    # =====================================================
    # 3. LAG FEATURES
    # =====================================================

    daily = create_lag_features(
        daily,
        target_col="jumlah_pasien",
        lags=(1, 7, 14, 21, 30),
        dropna=True
    )

    return daily