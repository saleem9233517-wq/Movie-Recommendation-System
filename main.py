
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"

# Load movies
movies = pd.read_csv(DATASET_DIR / "movies.csv")

print("Loading movie recommendation model...")

# Prepare genres for TF-IDF
movies["genres"] = movies["genres"].fillna("").str.replace(
    "|", " ", regex=False
)

# Convert genres into numerical features
tfidf = TfidfVectorizer()
genre_matrix = tfidf.fit_transform(movies["genres"])

# Calculate movie similarity
similarity_matrix = cosine_similarity(genre_matrix)

print("Model trained successfully!")
print(f"Total movies: {len(movies)}")


def recommend_movies(movie_title, number=5):
    # Find the selected movie
    matches = movies[
        movies["title"].str.contains(
            movie_title, case=False, regex=False, na=False
        )
    ]

    if matches.empty:
        print(f"\nMovie not found: {movie_title}")
        print("Please enter a movie title from the dataset.")
        return

    movie_index = matches.index[0]

    # Get similarity scores
    scores = list(enumerate(similarity_matrix[movie_index]))

    # Sort from most similar to least similar
    scores = sorted(scores, key=lambda item: item[1], reverse=True)

    # Exclude the selected movie itself
    scores = [
        item for item in scores
        if item[0] != movie_index
    ][:number]

    print(f"\nRecommendations for: {movies.iloc[movie_index]['title']}")
    print("-" * 55)

    for rank, (index, score) in enumerate(scores, start=1):
        title = movies.iloc[index]["title"]
        genres = movies.iloc[index]["genres"].replace(" ", ", ")

        print(f"{rank}. {title}")
        print(f"   Genres: {genres}")
        print(f"   Similarity: {score:.2%}\n")


# Example recommendation
recommend_movies("Toy Story", 5)
