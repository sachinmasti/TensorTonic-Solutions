import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    # Write code here
    X = np.array(X, dtype=float)  # ensure float for NaN handling
    if X.ndim == 1:  # handle 1D case
        mask = ~np.isnan(X)
        if np.any(mask):
            stat = np.mean(X[mask]) if strategy == 'mean' else np.median(X[mask])
        else:
            stat = 0.0
        X[~mask] = stat
        return X
    
    # handle 2D case
    X_out = X.copy()
    for col in range(X.shape[1]):
        mask = ~np.isnan(X[:, col])
        if np.any(mask):
            stat = np.mean(X[mask, col]) if strategy == 'mean' else np.median(X[mask, col])
        else:
            stat = 0.0
        X_out[~mask, col] = stat
    return X_out