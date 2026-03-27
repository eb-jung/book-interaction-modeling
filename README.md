# Book Interaction Modeling

A multi-task machine learning system for modeling user–book interactions on a Goodreads-style platform. The project covers three complementary prediction problems: **read prediction**, **genre classification from review text**, and **collaborative filtering for rating prediction**.

---

## Problem Statement

User–book interaction data is rich but noisy. A good recommender system needs to answer at least three questions:

1. **Would a user read this book at all?** (binary classification)  
2. **What genre does this book belong to?** (text classification from review)  
3. **How highly would a user rate this book?** (regression)

This project builds a separate model for each task, using collaborative filtering signals, NLP feature engineering, and matrix factorisation — then compares each against a simple baseline.

---

## Tasks and Modeling Approach

### 1. Read Prediction

**Goal:** Predict whether a given (user, book) pair represents a book the user would read (1) or not (0).

**Baseline:** Return 1 for books that account for the top 50% of total training interactions (popularity threshold).

**Model:** A logistic regression classifier trained on four hand-crafted collaborative-filtering features per (user, book) pair:

| Feature | Description |
|---------|-------------|
| Book popularity | Raw reader count — popular books are more likely read by anyone |
| Max Jaccard similarity | Maximum Jaccard overlap between the target book's readers and the readers of each book in the user's history |
| Mean Jaccard similarity | Average Jaccard across the user's full reading history |
| User history length | Number of books the user has already read |

Training uses balanced positive/negative samples — each observed (user, book) pair is paired with a randomly sampled unread book as a negative.

**Why it works:** The Jaccard similarity features capture community-level signals: if the users who read book A heavily overlap with the users who read book B, and a given user has read book A, they are a strong candidate to read book B. This goes beyond pure popularity and personalises predictions per user.

---

### 2. Category Prediction

**Goal:** Predict the genre of a book (5 classes) from the text of a user's review.

**Categories:** Children's · Comics/Graphic Novels · Fantasy/Paranormal · Mystery/Thriller/Crime · Young Adult

**Baseline:** Keyword matching — if the word "fantasy" appears → Fantasy, "mystery" → Mystery/Thriller, etc.

**Model:** TF-IDF vectorisation followed by a LinearSVC classifier.

- Unigrams + bigrams, sublinear TF scaling, IDF weighting  
- Up to 100k vocabulary features, filtered by document frequency (min_df=3, max_df=0.9)  
- Regularisation strength C selected by stratified 80/20 validation split  
- Final model retrained on the full training corpus

**Why it works:** Keyword matching only fires when reviewers explicitly name the genre. TF-IDF + SVC learns discriminative vocabulary patterns across all five categories — including indirect signals like character archetypes, setting descriptions, and plot vocabulary — which generalises far better to reviews that don't use obvious genre words.

---

### 3. Rating Prediction

**Goal:** Predict a user's star rating (1–5) for a book they have read, minimising MSE.

**Baseline:** Return the user's historical average rating (global average for unseen users).

**Model:** Regularised matrix factorisation (MF) with explicit bias terms.

The predicted rating for user *u* and book *i* is:

```
r̂(u,i) = μ + b_u + b_i + p_u · q_i
```

where μ is the global mean, b_u and b_i are learned scalar biases, and p_u, q_i are K-dimensional latent factor vectors updated by SGD with L2 regularisation and a decaying learning rate.

Hyperparameter selection (K ∈ {40, 60, 80}, lr ∈ {0.0025, 0.0035, 0.0045}) is performed on a held-out 20% validation split. The best configuration is retrained on the full dataset.

Cold-start handling:
- Unknown user, known book → μ + b_i  
- Known user, unknown book → μ + b_u  
- Both unknown → μ

**Why it works:** User-average baselines cannot distinguish between a user's varying affinity for different books. MF jointly learns that some books are universally rated lower (high b_i penalty) and that certain user–book latent factor alignments predict high ratings — which substantially reduces residual error beyond bias-only models.

---

## Repository Structure

```
book-interaction-modeling/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md               # Data field descriptions and setup instructions
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py           # Shared I/O: file readers, interaction maps, prediction writer
│   ├── read_prediction.py      # Jaccard feature engineering + logistic regression
│   ├── category_prediction.py  # TF-IDF vectorisation + LinearSVC genre classifier
│   ├── rating_prediction.py    # Matrix factorisation with user/item biases
│   └── evaluation.py           # Metrics, baselines, and baseline vs model comparison
│
├── scripts/
│   └── generate_predictions.py # End-to-end pipeline: trains all models and writes outputs
│
└── outputs/                    # Prediction CSVs written here (git-ignored)
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place data files

Put the following files in the `data/` directory (see `data/README.md` for field descriptions):

```
data/
  train_Interactions.csv.gz
  train_Category.json.gz
  test_Category.json.gz
  pairs_Read.csv
  pairs_Category.csv
  pairs_Rating.csv
```

### 3. Generate all predictions

```bash
python scripts/generate_predictions.py
```

This trains all three models and writes prediction files to `outputs/`.

### 4. Run specific tasks

```bash
# Read prediction only
python scripts/generate_predictions.py --tasks read

# Category and rating only
python scripts/generate_predictions.py --tasks category rating
```

---

## Baseline vs Model Summary

| Task | Baseline Approach | Model Approach | Improvement |
|------|-------------------|----------------|-------------|
| Read Prediction | Popularity threshold (top 50% of reads → 1) | Jaccard similarity features + logistic regression | Personalised per-user predictions using collaborative signals |
| Category Prediction | Keyword matching on genre words | TF-IDF (unigrams + bigrams) + LinearSVC | Learns discriminative vocabulary beyond explicit genre terms |
| Rating Prediction | User average (global average for cold-start) | Matrix factorisation with user/item biases + latent factors | Captures item-specific biases and user–item affinity |

---

## Possible Future Improvements

- **Read prediction:** Replace logistic regression with a gradient boosting model (XGBoost/LightGBM); add temporal features (recency of user interactions).
- **Category prediction:** Fine-tune a pre-trained sentence transformer (e.g., `all-MiniLM`) on review embeddings for better representation of long reviews.
- **Rating prediction:** Incorporate review text into the MF model (e.g., HFT or NARRE-style hybrid); add temporal dynamics (time-aware MF).
- **Shared:** Build a unified evaluation notebook with learning curves, confusion matrices, and error analysis by user/item cold-start degree.

---

## Tech Stack

- **Python 3.10+**
- **scikit-learn** — TF-IDF, LinearSVC, LogisticRegression, model selection
- **NumPy** — matrix factorisation, SGD updates
