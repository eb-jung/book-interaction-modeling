# Data

The data files used in this project are not included in the repository due to size. Place the following files in this directory before running the pipeline.

## Required Files

| File | Description |
|------|-------------|
| `train_Interactions.csv.gz` | User–book interaction records used for read and rating prediction |
| `train_Category.json.gz` | Labeled review records used for genre classification training |
| `test_Category.json.gz` | Unlabeled review records for category prediction inference |
| `pairs_Read.csv` | (user, book) pairs to predict read probability for |
| `pairs_Category.csv` | (user, review) pairs to predict genre for |
| `pairs_Rating.csv` | (user, book) pairs to predict star rating for |

## Data Formats

**`train_Interactions.csv.gz`**
```
userID,bookID,rating
u67805239,b61372131,4
```

**`train_Category.json.gz`** — one JSON record per line
```json
{"user_id": "u75242413", "review_id": "r45843137", "rating": 4,
 "review_text": "...", "genre": "mystery_thriller_crime", "genreID": 3}
```

**`pairs_*.csv`** — prediction target pairs with an empty prediction column
```
userID,bookID,prediction
u37758667,b99713185,
```

## Genre Label Map

| genreID | Genre |
|---------|-------|
| 0 | Children's |
| 1 | Comics / Graphic Novels |
| 2 | Fantasy / Paranormal |
| 3 | Mystery / Thriller / Crime |
| 4 | Young Adult / Romance |
