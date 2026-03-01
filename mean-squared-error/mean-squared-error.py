import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    return np.square(np.subtract(y_pred,y_true)).mean()
