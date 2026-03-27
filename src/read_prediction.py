"""
read_prediction.py
------------------
Predicts whether a user would read a given book (binary classification).

Approach
--------
Rather than relying solely on book popularity (the baseline strategy), we
compute a feature vector for each (user, book) pair that captures:

  1. Book popularity       — raw reader count; popular books are more likely
                             to be read by anyone.
  2. Max Jaccard similarity — maximum Jaccard overlap between the target book's
                             readers and the readers of any book in the user's
                             history. High overlap signals a strong community
                             signal for this user-book pair.
  3. Mean Jaccard similarity — average of the above over the user's full history;
                              smooths out noise from a single nearest neighbor.
  4. User history length   — how many books the user has already read; a proxy
                             for how active/explorable the user is.

These four features are fed into a logistic regression classifier trained on
balanced positive/negative samples drawn from the training interactions.

Baseline comparison
-------------------
The provided baseline returns 1 for the top-50%-by-interaction books and 0
otherwise. Our model personalizes predictions per user using collaborative
signals derived from Jaccard similarity.
"""

import random
import numpy as np
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets. Returns 0 if either is empty."""
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a) + len(set_b) - inter
    return inter / union if union > 0 else 0.0


def compute_features(
    user_id: str,
    book_id: str,
    readers_per_book: dict,
    books_per_user: dict,
) -> np.ndarray:
    """
    Return a 4-dimensional feature vector for a (user, book) pair.

    Features: [popularity, max_jaccard, mean_jaccard, user_history_len]
    """
    target_readers = readers_per_book.get(book_id, set())
    user_history = books_per_user.get(user_id, [])

    popularity = len(target_readers)
    history_len = len(user_history)

    jaccard_scores = []
    for prev_book in user_history:
        prev_readers = readers_per_book.get(prev_book)
        if prev_readers:
            jaccard_scores.append(_jaccard(target_readers, prev_readers))

    max_jaccard = max(jaccard_scores) if jaccard_scores else 0.0
    mean_jaccard = (sum(jaccard_scores) / len(jaccard_scores)) if jaccard_scores else 0.0

    return np.array([float(popularity), max_jaccard, mean_jaccard, float(history_len)])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    readers_per_book: dict,
    books_per_user: dict,
    max_pos_per_user: int = 80,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train a logistic regression classifier on balanced positive/negative samples.

    For each user, we sample up to `max_pos_per_user` books they actually read
    (positives) and pair each with a randomly selected unread book (negatives).

    Parameters
    ----------
    readers_per_book  : book_id -> set of user_ids
    books_per_user    : user_id -> list of book_ids
    max_pos_per_user  : cap on positives per user to keep training tractable
    random_state      : random seed for reproducibility

    Returns
    -------
    Fitted LogisticRegression model
    """
    random.seed(random_state)
    all_books = list(readers_per_book.keys())

    X, y = [], []

    for user_id, history in books_per_user.items():
        if not history:
            continue
        positives = random.sample(history, min(len(history), max_pos_per_user))

        for book_id in positives:
            # Positive sample
            X.append(compute_features(user_id, book_id, readers_per_book, books_per_user))
            y.append(1)

            # Negative sample — a book the user has NOT read
            neg = random.choice(all_books)
            while neg in set(history):
                neg = random.choice(all_books)
            X.append(compute_features(user_id, neg, readers_per_book, books_per_user))
            y.append(0)

    clf = LogisticRegression(max_iter=500, random_state=random_state)
    clf.fit(np.array(X), np.array(y))
    return clf


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(
    clf: LogisticRegression,
    user_id: str,
    book_id: str,
    readers_per_book: dict,
    books_per_user: dict,
) -> int:
    """Return 0 or 1 prediction for a single (user, book) pair."""
    features = compute_features(user_id, book_id, readers_per_book, books_per_user)
    return int(clf.predict(features.reshape(1, -1))[0])
