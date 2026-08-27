import pandas as pd
import holidays


# ==========================================================
# EVENT KEYWORDS
# ==========================================================

EVENT_KEYWORDS = {
    "is_imlek": "Lunar New Year",
    "is_idulfitri": "Eid al-Fitr",
    "is_idulfitri_sec": "Eid al-Fitr Second Day",
    "is_iduladha": "Eid al-Adha",
    "is_waisak": "Vesak Day",
    "is_natal": "Christmas Day",
    "is_nyepi": "Day of Silence",
}


# ==========================================================
# WEEKEND
# ==========================================================

def add_weekend(df):
    """
    Menambahkan fitur is_weekend.
    """

    df = df.copy()

    df["tanggal"] = pd.to_datetime(
        df["tanggal"]
    )

    df["is_weekend"] = (
        df["tanggal"]
        .dt.weekday
        .isin([5, 6])
        .astype(int)
    )

    return df


# ==========================================================
# INDONESIA HOLIDAYS
# ==========================================================

def get_indonesia_holidays(df):
    """
    Mengambil seluruh hari libur Indonesia
    berdasarkan tahun yang terdapat pada data.
    """

    years = df["tanggal"].dt.year.unique()

    indo = holidays.Indonesia(
        years=years
    )

    holiday_df = pd.DataFrame(
        list(indo.items()),
        columns=[
            "tanggal",
            "holiday_name"
        ]
    )

    holiday_df["tanggal"] = pd.to_datetime(
        holiday_df["tanggal"]
    )

    return holiday_df


# ==========================================================
# GET EVENT DATES
# ==========================================================

def get_event_dates(
    holiday_df,
    event_name
):
    """
    Mengambil tanggal suatu event
    berdasarkan nama hari libur.
    """

    return holiday_df.loc[
        holiday_df["holiday_name"].str.contains(
            event_name,
            case=False,
            na=False
        ),
        "tanggal"
    ]


# ==========================================================
# ADD EVENT RANGE
# ==========================================================

def add_event_range(
    df,
    event_dates,
    column_name,
    days_before=7,
    days_after=7
):
    """
    Menambahkan fitur event dengan rentang
    sebelum dan sesudah hari H.
    """

    df = df.copy()

    df[column_name] = 0

    for date in event_dates:

        start = (
            date
            - pd.Timedelta(days=days_before)
        )

        end = (
            date
            + pd.Timedelta(days=days_after)
        )

        mask = (
            (df["tanggal"] >= start)
            &
            (df["tanggal"] <= end)
        )

        df.loc[
            mask,
            column_name
        ] = 1

    return df


# ==========================================================
# EVENT FEATURES
# ==========================================================

def add_event_features(
    df,
    holiday_df
):
    """
    Menambahkan seluruh fitur hari raya.
    """

    df = df.copy()

    for column_name, event_name in EVENT_KEYWORDS.items():

        event_dates = get_event_dates(
            holiday_df,
            event_name
        )

        df = add_event_range(
            df,
            event_dates,
            column_name
        )

    return df


# ==========================================================
# NATIONAL HOLIDAY
# ==========================================================

def add_national_holiday(
    df,
    holiday_df
):
    """
    Menambahkan fitur is_libur_nasional.
    """

    df = df.copy()

    df["is_libur_nasional"] = (
        df["tanggal"]
        .isin(
            holiday_df["tanggal"]
        )
        .astype(int)
    )

    return df


# ==========================================================
# FEATURE ENGINEERING PIPELINE
# ==========================================================

def feature_engineering(df):
    """
    Menjalankan seluruh proses feature engineering
    """

    df = df.copy()

    # ------------------------------------------------------
    # Pastikan tanggal bertipe datetime
    # ------------------------------------------------------

    df["tanggal"] = pd.to_datetime(
        df["tanggal"]
    )

    # ------------------------------------------------------
    # Hari libur Indonesia
    # ------------------------------------------------------

    holiday_df = get_indonesia_holidays(
        df
    )

    # ------------------------------------------------------
    # Weekend
    # ------------------------------------------------------

    df = add_weekend(
        df
    )

    # ------------------------------------------------------
    # Event hari raya
    # ------------------------------------------------------

    df = add_event_features(
        df,
        holiday_df
    )

    # ------------------------------------------------------
    # Hari libur nasional
    # ------------------------------------------------------

    df = add_national_holiday(
        df,
        holiday_df
    )

    return df.reset_index(
        drop=True
    )
