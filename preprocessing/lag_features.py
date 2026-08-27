import pandas as pd

def remove_outliers(
    df,
    target_col="jumlah_pasien"
):
    """
    Mendeteksi dan menghapus outlier menggunakan
    metode Interquartile Range (IQR).

    Parameters
    ----------
    df : pd.DataFrame
        Data historis.

    target_col : str
        Nama kolom target yang digunakan
        untuk mendeteksi outlier.

    Returns
    -------
    pd.DataFrame
        Data setelah outlier dihapus.
    """

    df = df.copy()

    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    df = df[
        (df[target_col] >= lower_bound)
        &
        (df[target_col] <= upper_bound)
    ].copy()

    return df.reset_index(drop=True)

def create_lag_features(
    df,
    target_col="jumlah_pasien",
    lags=(1, 7, 14, 21, 30),
    dropna=True,
    remove_outlier=True
):
    """
    Membuat fitur lag berdasarkan kolom target
    dan menghapus outlier menggunakan metode IQR.

    Parameters
    ----------
    df : pd.DataFrame
        Data historis.

    target_col : str
        Nama kolom target.

    lags : tuple
        Daftar nilai lag.

    dropna : bool
        Menghapus baris yang memiliki nilai NaN
        akibat proses shifting.

    remove_outlier : bool
        Menghapus outlier setelah pembuatan
        fitur lag.

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    # =====================================================
    # MEMBUAT FITUR LAG
    # =====================================================

    for lag in lags:

        df[f"lag_{lag}"] = (
            df[target_col].shift(lag)
        )

    # =====================================================
    # MENGHAPUS NILAI NaN
    # =====================================================

    if dropna:

        df = df.dropna().reset_index(drop=True)

    # =====================================================
    # MENGHAPUS OUTLIER
    # =====================================================

    if remove_outlier:

        df = remove_outliers(
            df,
            target_col=target_col
        )

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
