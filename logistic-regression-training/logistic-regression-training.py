import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float).reshape(-1)

    N, D = X.shape

    # initialize parameters
    w = np.zeros(D)
    b = 0.0

    for _ in range(steps):
        # forward pass
        z = X @ w + b
        p = _sigmoid(z)

        # gradients
        error = p - y
        dw = (X.T @ error) / N
        db = np.mean(error)

        # update
        w -= lr * dw
        b -= lr * db

    return w, float(b)