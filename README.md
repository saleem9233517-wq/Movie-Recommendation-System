# 🎬 CineMatch — Hybrid Movie Recommendation System

CineMatch is a Python-based movie recommendation web application built with Streamlit. It recommends similar movies using movie genres, user-submitted tags, ratings, and popularity information.

## Features

- Content-based movie recommendations
- TF-IDF text feature extraction
- Cosine similarity between movie profiles
- User tags and genre integration
- Bayesian weighted movie ratings
- Rating-count-based popularity score
- Top Rated Movies page
- Interactive Streamlit interface
- Adjustable recommendation count and minimum rating filters

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- MovieLens dataset

## Dataset

This project uses the MovieLens Latest Small dataset from GroupLens.

Official source: https://grouplens.org/datasets/movielens/latest/

Place the following CSV files inside the `datasets` folder:

- `movies.csv`
- `ratings.csv`
- `tags.csv`
- `links.csv`

## Installation

Install the required libraries:

```bash
python -m pip install pandas scikit-learn streamlit
```

## Run the Application

From the project root directory, run:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## How It Works

1. Loads movie titles, genres, ratings, and tags.
2. Combines genres and user tags into movie profiles.
3. Uses TF-IDF to represent movie profiles numerically.
4. Calculates cosine similarity between movie profiles.
5. Combines content similarity, weighted rating quality, and rating-count popularity into a hybrid ranking score.
6. Displays recommendations through an interactive web interface.

## Project Structure

```text
Movie-Recommendation-System/
├── datasets/
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
│   └── links.csv
├── app.py
├── main.py
└── README.md
```

## Limitations

The current system uses content similarity with rating-aware ranking. It does not yet implement a trained collaborative-filtering model. Recommendation weights are initial choices and should be evaluated using held-out data.

## Author

Developed as a machine learning internship project.
