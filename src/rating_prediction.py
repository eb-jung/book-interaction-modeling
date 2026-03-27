"""
rating_prediction.py
--------------------
Predicts star ratings (1–5) for (user, book) pairs.

Approach
--------
We implement regularised matrix factorisation (MF) with explicit user and item
bias terms. The predicted rating for user u and book i is:

    r̂(u, i) = μ + b_u + b_i + p_u · q_i

where:
  μ    — global mean rating
  b_u  — learned scalar bias for user u
  b_i  — learned scalar bias for book i
  p_u  — K-dimensional latent factor vector for user u
  q_i  — K-dimensional latent factor vector for book i

Parameters are updated via SGD with L2 regularisation and a decaying learning
rate. A grid search over (K, learning_rate) selects the best configuration
on a held-out 20% validation split before final training on the full dataset.

Cold-start handling
-------------------
  - New user, known item  → μ + b_i
  - Known user, new item  → μ + b_u
  - Both new             → μ

Baseline comparison
-------------------
The provided baseline returns the user's average rating (or the global average
for unseen users). Our MF model captures item-specific biases and latent
user-item affinity, which substantially reduces MSE.
"""

import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    interactions: list,
    K: int = 60,
    learning_rate: float = 0.004,
    regularisation: float = 0.02,
    epochs: int = 40,
    lr_decay: float = 0.97,
    random_state: int = 42,
) -> dict:
    """
    Train a matrix factorisation model with user and item biases.

    Parameters
    ----------
    interactions   : list of (user_id, book_id, rating) tuples
    K              : number of latent dimensions
    learning_rate  : initial SGD step size
    regularisation : L2 penalty strength applied to all parameters
    epochs         : number of full passes over the training data
    lr_decay       : multiplicative decay applied to learning_rate each epoch
    random_state   : seed for reproducibility

    Returns
    -------
    dict with keys: mu, bu, bi, P, Q, u2idx, i2idx
    """
    rng = np.random.default_rng(random_state)
    random.seed(random_state)

    users = sorted({u for u, _, _ in interactions})
    items = sorted({i for _, i, _ in interactions})
    u2idx = {u: k for k, u in enumerate(users)}
    i2idx = {i: k for k, i in enumerate(items)}

    U, I = len(users), len(items)
    mu = float(np.mean([r for _, _, r in interactions]))
    bu = np.zeros(U)
    bi = np.zeros(I)
    P = 0.02 * rng.standard_normal((U, K))
    Q = 0.02 * rng.standard_normal((I, K))

    order = list(range(len(interactions)))
    lr = learning_rate

    for _ in range(epochs):
        random.shuffle(order)
        for idx in order:
            u, i, r = interactions[idx]
            uu, ii = u2idx[u], i2idx[i]
            pred = mu + bu[uu] + bi[ii] + P[uu] @ Q[ii]
            err = r - pred

            bu[uu] += lr * (err - regularisation * bu[uu])
            bi[ii] += lr * (err - regularisation * bi[ii])

            grad_P = err * Q[ii] - regularisation * P[uu]
            grad_Q = err * P[uu] - regularisation * Q[ii]
            P[uu] += lr * grad_P
            Q[ii] += lr * grad_Q

        lr *= lr_decay

    return {
        "mu": mu, "bu": bu, "bi": bi,
        "P": P, "Q": Q,
        "u2idx": u2idx, "i2idx": i2idx,
    }


def grid_search(
    interactions: list,
    k_values: list = None,
    lr_values: list = None,
    regularisation: float = 0.02,
    epochs: int = 35,
    val_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple:
    """
    Select the best (K, learning_rate) pair via validation MSE.

    Returns
    -------
    (best_K, best_lr, best_mse)
    """
    if k_values is None:
        k_values = [40, 60, 80]
    if lr_values is None:
        lr_values = [0.0025, 0.0035, 0.0045]

    idx = np.arange(len(interactions))
    tr_idx, va_idx = train_test_split(idx, test_size=val_size, random_state=random_state)
    tr = [interactions[i] for i in tr_idx]
    va = [interactions[i] for i in va_idx]
    true_ratings = [r for _, _, r in va]

    best_mse, best_K, best_lr = float("inf"), k_values[0], lr_values[0]

    for K in k_values:
        for lr in lr_values:
            model = train(tr, K=K, learning_rate=lr, regularisation=regularisation, epochs=epochs)
            preds = [
                float(np.clip(predict(u, b, model), 1, 5))
                for u, b, _ in va
            ]
            mse = mean_squared_error(true_ratings, preds)
            if verbose:
                print(f"  K={K:3d}  lr={lr:.4f}  val_MSE={mse:.4f}")
            if mse < best_mse:
                best_mse, best_K, best_lr = mse, K, lr

    if verbose:
        print(f"  → Best: K={best_K}, lr={best_lr:.4f}, val_MSE={best_mse:.4f}")

    return best_K, best_lr, best_mse


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(user_id: str, book_id: str, model: dict) -> float:
    """
    Predict the rating for a (user, book) pair.

    Handles cold-start users and items by falling back to bias-only predictions.
    Output is NOT clipped here — clip to [1, 5] at the call site if needed.
    """
    mu = model["mu"]
    u2idx, i2idx = model["u2idx"], model["i2idx"]
    bu, bi, P, Q = model["bu"], model["bi"], model["P"], model["Q"]

    known_u = user_id in u2idx
    known_i = book_id in i2idx

    if not known_u and not known_i:
        return mu
    if not known_u:
        return mu + bi[i2idx[book_id]]
    if not known_i:
        return mu + bu[u2idx[user_id]]

    uu, ii = u2idx[user_id], i2idx[book_id]
    return mu + bu[uu] + bi[ii] + P[uu] @ Q[ii]
