import pandas as pd

# Urutan hari dan bulan agar hasil tidak alfabet
DAY_ORDER = [
    "Senin",
    "Selasa",
    "Rabu",
    "Kamis",
    "Jumat",
    "Sabtu",
    "Minggu",
]

MONTH_ORDER = [
    "Januari",
    "Februari",
    "Maret",
    "April",
    "Mei",
    "Juni",
    "Juli",
    "Agustus",
    "September",
    "Oktober",
    "November",
    "Desember",
]


def validate_dataframe(df: pd.DataFrame):
    """
    Memastikan dataframe memiliki kolom yang diperlukan.
    """

    required_columns = ["tanggal", "jumlah_pasien"]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Kolom berikut tidak ditemukan: {missing}"
        )


def prepare_analysis_data(df: pd.DataFrame):
    """
    Menambahkan informasi hari, bulan, dan tahun.
    """

    validate_dataframe(df)

    data = df.copy()

    data["tanggal"] = pd.to_datetime(data["tanggal"])

    data["hari"] = data["tanggal"].dt.day_name()

    day_mapping = {
        "Monday": "Senin",
        "Tuesday": "Selasa",
        "Wednesday": "Rabu",
        "Thursday": "Kamis",
        "Friday": "Jumat",
        "Saturday": "Sabtu",
        "Sunday": "Minggu",
    }

    data["hari"] = data["hari"].map(day_mapping)

    data["bulan"] = data["tanggal"].dt.month_name()

    month_mapping = {
        "January": "Januari",
        "February": "Februari",
        "March": "Maret",
        "April": "April",
        "May": "Mei",
        "June": "Juni",
        "July": "Juli",
        "August": "Agustus",
        "September": "September",
        "October": "Oktober",
        "November": "November",
        "December": "Desember",
    }

    data["bulan"] = data["bulan"].map(month_mapping)

    data["tahun"] = data["tanggal"].dt.year

    return data


def average_by_day(df: pd.DataFrame):
    """
    Menghitung rata-rata pasien berdasarkan hari.
    """

    data = prepare_analysis_data(df)

    result = (
        data.groupby("hari")["jumlah_pasien"]
        .mean()
        .reindex(DAY_ORDER)
        .reset_index()
    )

    result.rename(
        columns={
            "jumlah_pasien": "rata_rata"
        },
        inplace=True,
    )

    return result


def average_by_month(df: pd.DataFrame):
    """
    Menghitung rata-rata pasien berdasarkan bulan.
    """

    data = prepare_analysis_data(df)

    result = (
        data.groupby("bulan")["jumlah_pasien"]
        .mean()
        .reindex(MONTH_ORDER)
        .reset_index()
    )

    result.rename(
        columns={
            "jumlah_pasien": "rata_rata"
        },
        inplace=True,
    )

    return result


def total_by_year(df: pd.DataFrame):
    """
    Menghitung total pasien tiap tahun.
    """

    data = prepare_analysis_data(df)

    result = (
        data.groupby("tahun")["jumlah_pasien"]
        .sum()
        .reset_index()
    )

    result.rename(
        columns={
            "jumlah_pasien": "total_pasien"
        },
        inplace=True,
    )

    return result


def busiest_day(df: pd.DataFrame):
    """
    Hari dengan rata-rata pasien tertinggi.
    """

    daily = average_by_day(df)

    row = daily.loc[
        daily["rata_rata"].idxmax()
    ]

    return {
        "hari": row["hari"],
        "rata_rata": round(row["rata_rata"], 2),
    }


def busiest_month(df: pd.DataFrame):
    """
    Bulan dengan rata-rata pasien tertinggi.
    """

    monthly = average_by_month(df)

    row = monthly.loc[
        monthly["rata_rata"].idxmax()
    ]

    return {
        "bulan": row["bulan"],
        "rata_rata": round(row["rata_rata"], 2),
    }


def busiest_year(df: pd.DataFrame):
    """
    Tahun dengan total pasien tertinggi.
    """

    yearly = total_by_year(df)

    row = yearly.loc[
        yearly["total_pasien"].idxmax()
    ]

    return {
        "tahun": int(row["tahun"]),
        "total_pasien": int(row["total_pasien"]),
    }


def analyze_patient_trends(df: pd.DataFrame):
    """
    Fungsi utama analisis data historis.
    """

    validate_dataframe(df)

    return {
        "daily_average": average_by_day(df),
        "monthly_average": average_by_month(df),
        "yearly_total": total_by_year(df),
        "busiest_day": busiest_day(df),
        "busiest_month": busiest_month(df),
        "busiest_year": busiest_year(df),
    }