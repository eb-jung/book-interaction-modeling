"""
generate_predictions.py
-----------------------
End-to-end pipeline: trains all three models and writes prediction files.

Usage
-----
    python scripts/generate_predictions.py

    # Run only specific tasks:
    python scripts/generate_predictions.py --tasks read category
    python scripts/generate_predictions.py --tasks rating

Expected data layout
--------------------
    data/
      train_Interactions.csv.gz
      train_Category.json.gz
      test_Category.json.gz
      pairs_Read.csv
      pairs_Category.csv
      pairs_Rating.csv

Output files are written to outputs/.
"""

import argparse
import sys
import os
import numpy as np

# Allow imports from src/ when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import data_utils, read_prediction, category_prediction, rating_prediction

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------

DATA_DIR = "data"
OUT_DIR = "outputs"

INTERACTIONS = os.path.join(DATA_DIR, "train_Interactions.csv.gz")
CATEGORY_TRAIN = os.path.join(DATA_DIR, "train_Category.json.gz")
CATEGORY_TEST = os.path.join(DATA_DIR, "test_Category.json.gz")
PAIRS_READ = os.path.join(DATA_DIR, "pairs_Read.csv")
PAIRS_CATEGORY = os.path.join(DATA_DIR, "pairs_Category.csv")
PAIRS_RATING = os.path.join(DATA_DIR, "pairs_Rating.csv")

OUT_READ = os.path.join(OUT_DIR, "predictions_Read.csv")
OUT_CATEGORY = os.path.join(OUT_DIR, "predictions_Category.csv")
OUT_RATING = os.path.join(OUT_DIR, "predictions_Rating.csv")


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------

def run_read():
    print("\n[READ PREDICTION] Building interaction maps...")
    readers_per_book, books_per_user = data_utils.build_interaction_maps(INTERACTIONS)

    print("[READ PREDICTION] Training Jaccard + logistic regression model...")
    clf = read_prediction.train(readers_per_book, books_per_user)

    _, pairs = data_utils.read_pairs(PAIRS_READ)
    print(f"[READ PREDICTION] Generating predictions for {len(pairs):,} pairs...")

    preds = []
    for user_id, book_id in pairs:
        p = read_prediction.predict(clf, user_id, book_id, readers_per_book, books_per_user)
        preds.append((user_id, book_id, p))

    data_utils.write_predictions(PAIRS_READ, OUT_READ, preds)
    print(f"[READ PREDICTION] Written → {OUT_READ}")


def run_category():
    print("\n[CATEGORY PREDICTION] Loading training reviews...")
    training_records = list(data_utils.read_json_gz(CATEGORY_TRAIN))

    print(f"[CATEGORY PREDICTION] Training TF-IDF + SVC on {len(training_records):,} reviews...")
    vectorizer, clf = category_prediction.train(training_records)

    print("[CATEGORY PREDICTION] Generating predictions on test reviews...")
    preds = []
    for record in data_utils.read_json_gz(CATEGORY_TEST):
        user_id = record.get("user_id") or record.get("userID")
        review_id = record.get("review_id") or record.get("reviewID")
        text = record.get("review_text", "")
        pred = category_prediction.predict(vectorizer, clf, text)
        preds.append((user_id, review_id, pred))

    data_utils.write_predictions(PAIRS_CATEGORY, OUT_CATEGORY, preds)
    print(f"[CATEGORY PREDICTION] Written → {OUT_CATEGORY}")


def run_rating():
    print("\n[RATING PREDICTION] Loading interaction data...")
    interactions = list(data_utils.read_interactions(INTERACTIONS))

    print(f"[RATING PREDICTION] Running grid search over K and learning rate...")
    best_K, best_lr, best_mse = rating_prediction.grid_search(interactions, verbose=True)

    print(f"\n[RATING PREDICTION] Retraining on full dataset (K={best_K}, lr={best_lr:.4f})...")
    model = rating_prediction.train(
        interactions, K=best_K, learning_rate=best_lr, regularisation=0.02, epochs=60
    )

    _, pairs = data_utils.read_pairs(PAIRS_RATING)
    print(f"[RATING PREDICTION] Generating predictions for {len(pairs):,} pairs...")

    preds = []
    for user_id, book_id in pairs:
        raw = rating_prediction.predict(user_id, book_id, model)
        clipped = float(np.clip(raw, 1, 5))
        preds.append((user_id, book_id, clipped))

    data_utils.write_predictions(PAIRS_RATING, OUT_RATING, preds)
    print(f"[RATING PREDICTION] Written → {OUT_RATING}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate predictions for all three book interaction modeling tasks."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["read", "category", "rating"],
        default=["read", "category", "rating"],
        help="Which tasks to run (default: all three).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    args = parse_args()
    task_set = set(args.tasks)

    if "read" in task_set:
        run_read()
    if "category" in task_set:
        run_category()
    if "rating" in task_set:
        run_rating()

    print("\nDone. Prediction files are in outputs/")
