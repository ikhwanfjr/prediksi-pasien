import numpy as np


def create_sequence(
    feature_matrix,
    time_steps=30
):
    """
    Membentuk sequence terakhir untuk input model GRU.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Matriks fitur hasil preprocessing
        dengan bentuk (n_samples, n_features).

    time_steps : int, default=30
        Jumlah time steps yang digunakan model.

    Returns
    -------
    np.ndarray
        Shape:
        (1, time_steps, n_features)
    """

    if len(feature_matrix) < time_steps:
        raise ValueError(
            f"Minimal membutuhkan {time_steps} baris data."
        )

    sequence = feature_matrix[-time_steps:]

    return np.expand_dims(
        sequence,
        axis=0
    )


def update_sequence(
    sequence,
    new_feature_row
):
    """
    Memperbarui sequence setelah memperoleh
    satu hasil prediksi baru.

    Parameters
    ----------
    sequence : np.ndarray
        Shape:
        (1, time_steps, n_features)

    new_feature_row : np.ndarray
        Shape:
        (n_features,)
        atau
        (1, n_features)

    Returns
    -------
    np.ndarray
        Sequence baru dengan shape yang sama.
    """

    sequence = sequence.copy()

    if new_feature_row.ndim == 1:
        new_feature_row = np.expand_dims(
            new_feature_row,
            axis=0
        )

    sequence = np.concatenate(
        [
            sequence[:, 1:, :],
            np.expand_dims(new_feature_row, axis=0)
        ],
        axis=1
    )

    return sequence