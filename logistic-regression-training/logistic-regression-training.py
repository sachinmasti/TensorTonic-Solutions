import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    N, features = X.shape
    w = np.zeros(features)  # weights 0 se start
    b = 0                   # bias 0 se start

    for i in range(steps):
        # Step 1: Predict karo
        p = _sigmoid(X @ w + b)

        # Step 2: Gradient nikalo (kitna adjust karna hai)
        dw = (1/N) * X.T @ (p - y)
        db = (1/N) * np.sum(p - y)

        # Step 3: Weights update karo
        w = w - lr * dw
        b = b - lr * db

    return w, b