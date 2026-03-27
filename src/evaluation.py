"""
evaluation.py
-------------
Evaluation utilities and baseline implementations for comparison.

Provides:
  - Accuracy and MSE metric helpers
  - Popularity-threshold baseline for read prediction
  - Keyword-lookup baseline for category prediction
  - User-average baseline for rating prediction
"""

from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def classification_accuracy(true_labels, predicted_labels) -> float:
    """Fraction of correctly predicted labels."""
    return accuracy_score(true_labels, predicted_labels)


def rating_mse(true_ratings, predicted_ratings) -> float:
    """Mean squared error between predicted and true ratings."""
    return mean_squared_error(true_ratings, predicted_ratings)


# ---------------------------------------------------------------------------
# Baselines (mirroring baselines.py logic, wrapped as callable functions)
# ---------------------------------------------------------------------------

def popularity_threshold_baseline(
    pairs: list,
    readers_per_book: dict,
    threshold: float = 0.5,
) -> list:
    """
    Read prediction baseline: return 1 for books that account for
    the top `threshold` fraction of total interactions.

    This mirrors the provided baselines.py implementation.

    Parameters
    ----------
    pairs            : list of (user_id, book_id) tuples
    readers_per_book : book_id -> set of reader user_ids
    threshold        : popularity cutoff fraction (default 0.5)

    Returns
    -------
    list of integer predictions (0 or 1)
    """
    book_counts = {b: len(readers) for b, readers in readers_per_book.items()}
    total_reads = sum(book_counts.values())
    sorted_books = sorted(book_counts, key=book_counts.get, reverse=True)

    popular = set()
    cumulative = 0
    for book in sorted_books:
        cumulative += book_counts[book]
        popular.add(book)
        if cumulative > total_reads * threshold:
            break

    return [1 if book_id in popular else 0 for _, book_id in pairs]


def keyword_category_baseline(review_texts: list) -> list:
    """
    Category prediction baseline: classify by presence of genre keywords.

    This mirrors the provided baselines.py implementation.

    Parameters
    ----------
    review_texts : list of raw review strings

    Returns
    -------
    list of integer genreID predictions
    """
    keyword_rules = [
        (0, ["children"]),
        (1, ["comic", "graphic"]),
        (2, ["fantasy"]),
        (3, ["mystery"]),
        (4, ["love", "romance"]),
    ]
    predictions = []
    for text in review_texts:
        lower = text.lower()
        pred = 2  # default: fantasy (most common class)
        for genre_id, keywords in keyword_rules:
            if any(kw in lower for kw in keywords):
                pred = genre_id
                break
        predictions.append(pred)
    return predictions


def user_average_rating_baseline(
    pairs: list,
    interactions: list,
) -> list:
    """
    Rating baseline: return the user's average rating, or the global
    average for unseen users.

    This mirrors the provided baselines.py implementation.

    Parameters
    ----------
    pairs        : list of (user_id, book_id) tuples
    interactions : list of (user_id, book_id, rating) tuples

    Returns
    -------
    list of float rating predictions
    """
    user_ratings = defaultdict(list)
    all_ratings = []
    for u, _, r in interactions:
        user_ratings[u].append(r)
        all_ratings.append(r)

    global_avg = np.mean(all_ratings)
    user_avg = {u: np.mean(rs) for u, rs in user_ratings.items()}

    return [
        user_avg.get(user_id, global_avg)
        for user_id, _ in pairs
    ]


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_methods(task: str, baseline_preds, model_preds, true_labels):
    """
    Print a side-by-side comparison of baseline vs model performance.

    Parameters
    ----------
    task          : short label for the task ("read", "category", "rating")
    baseline_preds: predictions from the baseline method
    model_preds   : predictions from the improved model
    true_labels   : ground-truth labels or ratings
    """
    if task == "rating":
        baseline_score = rating_mse(true_labels, baseline_preds)
        model_score = rating_mse(true_labels, model_preds)
        metric_name = "MSE (lower is better)"
        improved = model_score < baseline_score
    else:
        baseline_score = classification_accuracy(true_labels, baseline_preds)
        model_score = classification_accuracy(true_labels, model_preds)
        metric_name = "Accuracy (higher is better)"
        improved = model_score > baseline_score

    improvement_sign = "✓" if improved else "✗"
    print(f"\n[{task.upper()}] {metric_name}")
    print(f"  Baseline : {baseline_score:.4f}")
    print(f"  Model    : {model_score:.4f}  {improvement_sign}")
