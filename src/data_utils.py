"""
data_utils.py
-------------
Shared I/O helpers for reading Goodreads interaction data.
All other modules import from here to keep data loading consistent.
"""

import ast
import gzip
from collections import defaultdict


# ---------------------------------------------------------------------------
# Raw file readers
# ---------------------------------------------------------------------------

def read_interactions(path: str):
    """
    Yield (user_id, book_id, rating) tuples from a gzipped CSV interaction file.
    Skips the header row automatically.
    """
    with gzip.open(path, "rt") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            yield parts[0], parts[1], int(parts[2])


def read_json_gz(path: str):
    """
    Yield Python dicts from a gzipped JSON-lines file (one dict per line).
    Uses ast.literal_eval for safety instead of bare eval().
    """
    with gzip.open(path, "rt") as f:
        for line in f:
            yield ast.literal_eval(line)


def read_pairs(path: str):
    """
    Read a pairs CSV (userID, bookID or userID, reviewID).
    Returns the header line and a list of (col1, col2) tuples.
    """
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        pairs = [tuple(line.strip().split(",")) for line in f]
    return header, pairs


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------

def build_interaction_maps(interactions_path: str):
    """
    Build two lookup structures from the training interaction file:

    - readers_per_book : book_id -> set of user_ids who read it
    - books_per_user   : user_id -> list of book_ids the user read

    These are used by both read prediction and rating prediction.
    """
    readers_per_book = defaultdict(set)
    books_per_user = defaultdict(list)

    for user_id, book_id, _ in read_interactions(interactions_path):
        readers_per_book[book_id].add(user_id)
        books_per_user[user_id].append(book_id)

    return dict(readers_per_book), dict(books_per_user)


# ---------------------------------------------------------------------------
# Prediction file writer
# ---------------------------------------------------------------------------

def write_predictions(pairs_path: str, out_path: str, predictions):
    """
    Write predictions to a CSV file that mirrors the pairs file format.

    Parameters
    ----------
    pairs_path  : path to the input pairs CSV (to copy the header)
    out_path    : path to write predictions
    predictions : iterable of (col1, col2, prediction) tuples
    """
    with open(pairs_path, "r", encoding="utf-8") as f_pairs:
        header = f_pairs.readline().strip()

    cols = [c.strip().lower() for c in header.split(",")]
    if "prediction" not in cols:
        header = header + ",prediction"

    with open(out_path, "w", encoding="utf-8") as f_out:
        f_out.write(header + "\n")
        for c1, c2, pred in predictions:
            f_out.write(f"{c1},{c2},{pred}\n")
