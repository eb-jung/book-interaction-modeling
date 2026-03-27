"""
category_prediction.py
----------------------
Predicts the genre/category of a book from its review text (5-class classification).

Categories
----------
  0 — Children's
  1 — Comics / Graphic Novels
  2 — Fantasy / Paranormal
  3 — Mystery / Thriller / Crime
  4 — Young Adult / Romance

Approach
--------
We represent each review as a TF-IDF vector over unigrams and bigrams (up to
100k features, sublinear TF scaling, standard IDF). A LinearSVC is then trained
on these vectors. A small C grid search is performed on a stratified 80/20 split,
and the best model is retrained on the full training corpus before inference.

Baseline comparison
-------------------
The provided baseline uses a simple keyword lookup (e.g., if "fantasy" appears →
predict Fantasy). Our TF-IDF + SVC approach learns discriminative features
across the full vocabulary and generalises far better to reviews that don't
contain obvious genre keywords.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

# Human-readable genre names (index = genreID)
GENRE_NAMES = {
    0: "children",
    1: "comics_graphic",
    2: "fantasy_paranormal",
    3: "mystery_thriller_crime",
    4: "young_adult",
}

# Fallback keyword → genreID mapping used only when the dataset label is absent
_KEYWORD_MAP = [
    (3, ["mystery", "thriller", "crime"]),
    (1, ["comic", "graphic"]),
    (0, ["child"]),
    (4, ["romance", "romantic", "young adult", "ya"]),
    (2, ["fantasy", "magic", "paranormal"]),
]


def _label_from_raw(raw_genre: str) -> int:
    """Map a raw genre string to a genreID via keyword matching."""
    s = (raw_genre or "").lower()
    for genre_id, keywords in _KEYWORD_MAP:
        if any(kw in s for kw in keywords):
            return genre_id
    return 2  # default: fantasy (most common category)


def extract_label(record: dict) -> int:
    """
    Extract a numeric genre label from a training record.
    Prefers the explicit `genreID` field; falls back to keyword matching on
    the `genre` string field.
    """
    gid = record.get("genreID")
    if isinstance(gid, int) and 0 <= gid <= 4:
        return gid
    return _label_from_raw(record.get("genre") or record.get("category", ""))


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_vectorizer() -> TfidfVectorizer:
    """Return a TF-IDF vectorizer with settings tuned for review text."""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        max_features=100_000,
    )


def train(
    training_records,
    c_values: list = None,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train a TF-IDF + LinearSVC pipeline on review text.

    Parameters
    ----------
    training_records : iterable of dicts (from read_json_gz on train_Category.json.gz)
    c_values         : list of C regularisation strengths to search over
    val_size         : fraction of data to hold out for model selection
    random_state     : seed for reproducibility

    Returns
    -------
    (vectorizer, classifier) — both fitted on the full training set after
    model selection is complete.
    """
    if c_values is None:
        c_values = [0.5, 1.0]

    texts, labels = [], []
    for record in training_records:
        text = record.get("review_text", "")
        if not text:
            continue
        texts.append(text)
        labels.append(extract_label(record))

    labels = np.array(labels)
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(texts)

    # Stratified split for model selection
    idx = np.arange(len(labels))
    tr_idx, va_idx = train_test_split(
        idx, test_size=val_size, random_state=random_state, stratify=labels
    )

    best_acc, best_C = -1.0, c_values[0]
    for C in c_values:
        clf = LinearSVC(C=C, max_iter=2000)
        clf.fit(X[tr_idx], labels[tr_idx])
        acc = accuracy_score(labels[va_idx], clf.predict(X[va_idx]))
        if acc > best_acc:
            best_acc, best_C = acc, C

    # Retrain on full data with the best C
    final_clf = LinearSVC(C=best_C, max_iter=2000)
    final_clf.fit(X, labels)

    return vectorizer, final_clf


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(vectorizer: TfidfVectorizer, clf: LinearSVC, review_text: str) -> int:
    """Predict the genreID for a single review string."""
    X = vectorizer.transform([review_text])
    return int(clf.predict(X)[0])
