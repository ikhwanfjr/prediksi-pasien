import json
import joblib
import xgboost as xgb

from prophet.serialize import model_from_json

from tensorflow.keras.models import load_model


# =====================================================
# Path
# =====================================================

MODEL_PATH = "models"
FEATURE_PATH = "features"
SCALER_PATH = "scalers"


# =====================================================
# Prophet
# =====================================================

def load_prophet():
    """
    Memuat model Prophet beserta daftar regressornya.
    """

    with open(
        f"{MODEL_PATH}/prophet_model.json",
        "r"
    ) as f:

        prophet_model = model_from_json(
            f.read()
        )

    with open(
        f"{FEATURE_PATH}/prophet_regressors.json",
        "r"
    ) as f:

        prophet_regressors = json.load(f)

    return prophet_model, prophet_regressors


# =====================================================
# XGBoost
# =====================================================

def load_xgboost():
    """
    Memuat model XGBoost beserta daftar fitur.
    """

    model = xgb.XGBRegressor()

    model.load_model(
        f"{MODEL_PATH}/xgboost_model.json"
    )

    with open(
        f"{FEATURE_PATH}/features.json",
        "r"
    ) as f:

        feature_list = json.load(f)

    return model, feature_list


# =====================================================
# GRU
# =====================================================

def load_gru():
    """
    Memuat model GRU, scaler, dan daftar fitur.
    """

    model = load_model(
        f"{MODEL_PATH}/gru_model.keras"
    )

    scaler_X = joblib.load(
        f"{SCALER_PATH}/scaler_X.pkl"
    )

    scaler_y = joblib.load(
        f"{SCALER_PATH}/scaler_y.pkl"
    )

    with open(
        f"{FEATURE_PATH}/features.json",
        "r"
    ) as f:

        feature_list = json.load(f)

    return (
        model,
        scaler_X,
        scaler_y,
        feature_list,
    )