import pandas as pd


def create_lag_features(
    df,
    target_col="jumlah_pasien",
    lags=(1, 7, 14, 21, 30),
    dropna=True
):
    """
    Membuat fitur lag berdasarkan kolom target.

    Parameters
    ----------
    df : pd.DataFrame
        Data historis.
    target_col : str
        Nama kolom target.
    lags : tuple
        Daftar nilai lag.
    dropna : bool
        Menghapus baris yang memiliki nilai NaN akibat proses shifting.

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    if dropna:
        df = df.dropna().reset_index(drop=True)

    return df


def update_target_history(
    history,
    prediction,
    max_history=30
):
    """
    Menambahkan hasil prediksi ke riwayat target.

    Parameters
    ----------
    history : list
        Riwayat jumlah pasien.
    prediction : float
        Hasil prediksi terbaru.
    max_history : int
        Jumlah maksimum riwayat yang disimpan.

    Returns
    -------
    list
    """

    history = history.copy()

    history.append(float(prediction))

    if len(history) > max_history:
        history.pop(0)

    return history


def build_lag_features(
    history,
    lags=(1, 7, 14, 21, 30)
):
    """
    Membentuk fitur lag dari history terbaru.

    Parameters
    ----------
    history : list
        Riwayat jumlah pasien.

    Returns
    -------
    dict
        Contoh:
        {
            "lag_1": ...,
            "lag_7": ...,
            ...
        }
    """

    lag_features = {}

    for lag in lags:

        if len(history) < lag:
            raise ValueError(
                f"History minimal harus memiliki {lag} data."
            )

        lag_features[f"lag_{lag}"] = history[-lag]

    return lag_features