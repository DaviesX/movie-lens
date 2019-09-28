import numpy as np

def rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the mse loss.

    Arguments:
        y {[np.ndarray]} -- 1D float values.
        y_pred {[np.ndarray]} -- 1D float values.

    Returns:
        float -- MSE value.
    """
    d = (y_pred - y)
    return np.sqrt(np.mean(d*d))
