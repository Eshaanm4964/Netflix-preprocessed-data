# Netflix-preprocessed-data

# Netflix Movies and TV Shows Data Preprocessing

This repository contains a full preprocessing pipeline for the **Netflix Movies and TV Shows Dataset**, originally sourced from [Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows). The project involves data cleaning, textual preprocessing using NLP techniques, feature engineering, and exporting a ready-to-use dataset for machine learning or data analysis.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Dataset Used

**Filename:** `netflix_titles.csv`  
**Source:** Kaggle  
**Features:** Title, Cast, Country, Date Added, Duration, Description, Rating, etc.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📌 Objective

To clean and preprocess raw Netflix data and prepare it for:

- Exploratory Data Analysis (EDA)
- Natural Language Processing (NLP)
- Machine Learning model building

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Technologies & Libraries

- Python 3.x
- Pandas, NumPy
- NLTK (for NLP preprocessing)
- Scikit-learn (for encoding and vectorization)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

##  Preprocessing Steps

###  Missing Value Handling
- Filled `NaN` in `director`, `cast`, `country`, `rating` with `"Unknown"` or `"unrated"`.
- Filled `date_added` with a placeholder and then converted it to datetime.
- Replaced missing `duration` with `"mean"` as a placeholder.

### 🔸 Data Cleaning
- Converted `date_added` into `dd-mm-yyyy` format.
- Dropped `show_id` after label encoding the `type` column (`0 = Movie`, `1 = TV Show`).

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🧠 Textual Preprocessing (NLP)

Applied on: `title`, `cast`, `listed_in`, and `description`

- **Lowercasing**
- **Punctuation Removal**
- **Tokenization** (`word_tokenize`)
- **Stopword Removal** (`nltk.corpus.stopwords`)
- **Lemmatization** (`WordNetLemmatizer`)
- Final cleaned text saved in columns like `title_clean`, `description_clean`, etc.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🧬 Feature Engineering

- `description_clean` → Transformed using **TF-IDF** vectorization (500 features)
- `listed_in_clean` → Transformed using **MultiLabelBinarizer**
- `country`, `rating` → Transformed using **LabelEncoder**
- Extracted `genres` list from `listed_in_clean`

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📦 Final Exported Dataset

**Filename:** `netflix_final_preprocessed.csv`

### 🔑 Columns Included:
| Column             | Description                            |
|--------------------|----------------------------------------|
| `type`             | Binary target (0 = Movie, 1 = TV Show) |
| `title_clean`      | Cleaned title                          |
| `cast_clean`       | Cleaned cast members                   |
| `listed_in_clean`  | Cleaned genre string                   |
| `description_clean`| Cleaned plot description               |
| `genres`           | Tokenized genre list (for multi-label) |
| `release_year`     | Year of release                        |
| `duration`         | Duration text (e.g., "90 min")         |
| `country`          | Country of origin                      |
| `rating`           | Content rating                         |
| `country_encoded`  | Label-encoded country                  |
| `rating_encoded`   | Label-encoded rating                   |




