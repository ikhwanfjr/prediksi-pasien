import pandas as pd


def aggregate_daily(
    df: pd.DataFrame,
    admission_col: str = "tgl_masuk",
    discharge_col: str = "tgl_keluar"
) -> pd.DataFrame:
    """
    Membersihkan data rawat inap dan mengubahnya
    menjadi jumlah pasien per hari.
    """

    df = df.copy()

    # Hapus duplikat
    df = df.drop_duplicates().reset_index(drop=True)

    # Konversi tanggal
    df[admission_col] = pd.to_datetime(
        df[admission_col],
        errors="coerce"
    )

    df[discharge_col] = pd.to_datetime(
        df[discharge_col],
        errors="coerce"
    )

    # Hapus tanggal yang tidak valid
    df = df.dropna(
        subset=[
            admission_col,
            discharge_col
        ]
    )

    # Hapus data tidak konsisten
    # tanggal keluar tidak boleh sebelum tanggal masuk
    df = df[
        df[discharge_col] >= df[admission_col]
    ].reset_index(drop=True)

    # Membuat rentang tanggal setiap pasien
    df["tanggal"] = df.apply(
        lambda row: pd.date_range(
            start=row[admission_col],
            end=row[discharge_col],
            freq="D"
        ),
        axis=1
    )

    # Explode tanggal
    df = df.explode(
        "tanggal"
    ).reset_index(drop=True)

    # Agregasi jumlah pasien per hari
    daily = (
        df.groupby("tanggal")
        .size()
        .reset_index(
            name="jumlah_pasien"
        )
        .sort_values("tanggal")
        .reset_index(drop=True)
    )

    return daily