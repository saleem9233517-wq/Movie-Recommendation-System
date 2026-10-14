import html
import math
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"

# ---------------- DESIGN ----------------
st.markdown(
    """
    <style>
    /* Main page */
    .stApp {
        background-color: #0e1117;
        color: #f5f5f5;
    }

    /* Prevent Streamlit's top toolbar from overlapping the hero */
    [data-testid="stAppViewBlockContainer"] {
        padding-top: 3rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 30px 30px;
        border-radius: 18px;
        background: linear-gradient(120deg, #24243e, #302b63, #0f0c29);
        margin: 18px 0 30px 0;
        min-height: 155px;
        height: auto;
        box-sizing: border-box;
        overflow: visible;
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: clamp(1.8rem, 3vw, 2.5rem);
        line-height: 1.3 !important;
        margin: 0 0 12px 0 !important;
        padding: 0 !important;
        overflow-wrap: anywhere;
        white-space: normal;
    }

    .hero p {
        color: #e2e2f7 !important;
        font-size: 17px;
        line-height: 1.55;
        margin: 0 !important;
    }

    /* Improve labels and help text contrast */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stMarkdownContainer"] p,
    .stCaption {
        color: #e1e5ef;
    }

    [data-testid="stWidgetLabel"] p {
        font-weight: 600;
    }

    /* Recommendation cards */
    .movie-card {
        background: #1c2030;
        border: 1px solid #454b65;
        border-radius: 14px;
        padding: 20px;
        min-height: 205px;
        height: auto;
        margin-bottom: 14px;
        overflow-wrap: anywhere;
    }

    .movie-title {
        color: #c4b5fd;
        font-size: 18px;
        line-height: 1.4;
        font-weight: 700;
    }

    .movie-meta {
        color: #e1e5ef;
        font-size: 14px;
        line-height: 1.6;
        margin-top: 5px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f1f3f9;
    }

    section[data-testid="stSidebar"] * {
        color: #202538;
    }

    /* Buttons */
    .stButton button {
        border-radius: 9px;
        font-weight: 600;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #454b65;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    movies = pd.read_csv(DATASET_DIR / "movies.csv")
    ratings = pd.read_csv(DATASET_DIR / "ratings.csv")
    tags = pd.read_csv(DATASET_DIR / "tags.csv")

    movies["genres"] = movies["genres"].fillna("")

    # Average rating and number of ratings per movie
    rating_summary = (
        ratings.groupby("movieId")
        .agg(
            average_rating=("rating", "mean"),
            rating_count=("rating", "count"),
        )
        .reset_index()
    )

    movies = movies.merge(rating_summary, on="movieId", how="left")

    # Combine user-submitted tags for each movie
    tags["tag"] = tags["tag"].fillna("").astype(str)
    tag_summary = (
        tags.groupby("movieId")["tag"]
        .apply(lambda values: " ".join(values))
        .reset_index(name="user_tags")
    )
    movies = movies.merge(tag_summary, on="movieId", how="left")

    movies["user_tags"] = movies["user_tags"].fillna("")
    overall_rating = float(ratings["rating"].mean())

    movies["average_rating"] = movies["average_rating"].fillna(overall_rating)
    movies["rating_count"] = movies["rating_count"].fillna(0)

    # Bayesian weighted rating reduces the impact of very few ratings
    minimum_votes = 20
    denominator = movies["rating_count"] + minimum_votes
    movies["weighted_rating"] = (
        (movies["rating_count"] / denominator) * movies["average_rating"]
        + (minimum_votes / denominator) * overall_rating
    )
    movies["rating_score"] = movies["weighted_rating"] / 5.0

    # Log-scaled popularity score
    log_counts = movies["rating_count"].apply(lambda count: math.log1p(count))
    max_count = float(log_counts.max())
    if max_count > 0:
        movies["popularity_score"] = log_counts / max_count
    else:
        movies["popularity_score"] = 0.0

    # Combine genres and user tags into a text profile
    movies["movie_profile"] = (
        movies["genres"].str.replace("|", " ", regex=False)
        + " "
        + movies["user_tags"]
    ).str.strip()

    # Avoid an empty text profile causing TF-IDF to fail
    empty_profiles = movies["movie_profile"].str.strip().eq("")
    movies.loc[empty_profiles, "movie_profile"] = "unknown"

    return movies, ratings, tags


@st.cache_resource
def build_similarity(profiles):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(profiles)
    return cosine_similarity(matrix)


try:
    movies, ratings, tags = load_data()
    similarity_matrix = build_similarity(movies["movie_profile"].tolist())
except Exception as error:
    st.error(f"Could not load or process the datasets: {error}")
    st.info(
        "Check that movies.csv, ratings.csv and tags.csv are inside the "
        "project's datasets folder."
    )
    st.stop()


# ---------------- SIDEBAR ----------------
st.sidebar.title("🎬 CineMatch")
st.sidebar.caption("Hybrid Movie Recommendation System")

page = st.sidebar.radio(
    "Navigation",
    ["Discover Movies", "Top Rated", "About Project"],
)

st.sidebar.divider()
st.sidebar.metric("Movies Available", f"{len(movies):,}")
st.sidebar.metric("User Ratings", f"{len(ratings):,}")
st.sidebar.metric("User Tags", f"{len(tags):,}")


# ---------------- HERO HEADER ----------------
st.markdown(
    """
    <div class="hero">
        <h1>Find Your Next Favourite Movie 🍿</h1>
        <p>Discover movies using genres, user tags and rating information.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------- DISCOVER MOVIES ----------------
if page == "Discover Movies":
    st.subheader("🔎 Discover Similar Movies")

    movie_titles = sorted(movies["title"].dropna().unique().tolist())
    default_title = "Toy Story (1995)"
    default_index = movie_titles.index(default_title) if default_title in movie_titles else 0

    selected_movie = st.selectbox(
        "Choose a movie",
        movie_titles,
        index=default_index,
    )

    number = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5,
    )

    min_ratings = st.selectbox(
        "Minimum user ratings for recommended movies",
        [0, 5, 10, 20],
        index=0,
    )

    if st.button("✨ Recommend Movies", type="primary"):
        selected_indices = movies.index[movies["title"] == selected_movie]
        if len(selected_indices) == 0:
            st.warning("Selected movie could not be found.")
        else:
            movie_index = int(selected_indices[0])
            scores = similarity_matrix[movie_index]
            candidates = []

            for idx, similarity in enumerate(scores):
                if idx == movie_index:
                    continue

                movie = movies.iloc[idx]
                if movie["rating_count"] < min_ratings:
                    continue

                # Hybrid ranking:
                # 75% content similarity, 20% weighted rating,
                # 5% rating-count popularity
                hybrid_score = (
                    0.75 * float(similarity)
                    + 0.20 * float(movie["rating_score"])
                    + 0.05 * float(movie["popularity_score"])
                )

                candidates.append((idx, float(similarity), hybrid_score))

            candidates.sort(key=lambda item: item[2], reverse=True)
            recommendations = candidates[:number]

            if not recommendations:
                st.warning("No movies match these filters. Try a lower minimum rating count.")
            else:
                st.success(f"Recommendations for {selected_movie}")
                columns = st.columns(2)

                for rank, (idx, similarity, hybrid_score) in enumerate(
                    recommendations, start=1
                ):
                    movie = movies.iloc[idx]
                    safe_title = html.escape(str(movie["title"]))
                    safe_genres = html.escape(
                        str(movie["genres"]).replace("|", ", ")
                    )
                    rating_count = int(movie["rating_count"])
                    average_rating = float(movie["average_rating"])

                    with columns[(rank - 1) % 2]:
                        st.markdown(
                            f"""
                            <div class="movie-card">
                                <div class="movie-title">
                                    {rank}. {safe_title}
                                </div>
                                <div class="movie-meta">
                                    <b>Genres:</b> {safe_genres}
                                </div>
                                <div class="movie-meta">
                                    <b>Average user rating:</b>
                                    {average_rating:.2f} / 5
                                </div>
                                <div class="movie-meta">
                                    <b>Number of ratings:</b> {rating_count:,}
                                </div>
                                <div class="movie-meta">
                                    <b>Content similarity:</b> {similarity:.1%}
                                </div>
                                <div class="movie-meta">
                                    <b>Hybrid ranking score:</b> {hybrid_score:.1%}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                st.caption(
                    "The hybrid ranking combines content similarity, weighted "
                    "rating quality and rating-count popularity. It is a ranking "
                    "score, not a probability that you will enjoy a movie."
                )


# ---------------- TOP RATED ----------------
elif page == "Top Rated":
    st.subheader("⭐ Top Rated Movies")
    st.write(
        "Filter out movies with very few ratings to make the list more useful."
    )

    min_ratings_top = st.slider(
        "Minimum number of user ratings",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
    )

    top_movies = (
        movies[movies["rating_count"] >= min_ratings_top]
        .sort_values(
            ["weighted_rating", "rating_count"],
            ascending=[False, False],
        )
        .head(20)
    )

    display_movies = top_movies[
        ["title", "genres", "average_rating", "rating_count", "weighted_rating"]
    ].rename(
        columns={
            "title": "Movie",
            "genres": "Genres",
            "average_rating": "Average Rating (/5)",
            "rating_count": "Number of Ratings",
            "weighted_rating": "Weighted Rating (/5)",
        }
    )

    st.dataframe(display_movies, use_container_width=True, hide_index=True)


# ---------------- ABOUT PROJECT ----------------
elif page == "About Project":
    st.subheader("About CineMatch")

    st.markdown(
        """
        CineMatch is a hybrid movie recommendation project built with Python.

        **Data used**
        - Movie genres from `movies.csv`
        - User-submitted tags from `tags.csv`
        - User ratings from `ratings.csv`

        **How recommendations work**
        1. Genres and tags are combined into a text profile.
        2. TF-IDF converts text profiles into numerical features.
        3. Cosine similarity measures content similarity between movies.
        4. Average ratings are adjusted with a Bayesian weighted rating.
        5. Rating counts contribute a popularity score.
        6. These signals are combined into a hybrid ranking score.

        **Current limitation:** This is a content-based system with rating-aware
        ranking, not a trained collaborative-filtering model. The weights are
        initial choices and should be evaluated on held-out data before making
        claims about recommendation quality.
        """
    )
