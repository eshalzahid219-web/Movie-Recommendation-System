
import gradio as gr
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "ml-latest-small"

movies = pd.read_csv(DATA_DIR / "movies.csv")
ratings = pd.read_csv(DATA_DIR / "ratings.csv")

# Prepare movie and user indexes.
movie_ids = movies["movieId"].tolist()
movie_to_index = {movie_id: i for i, movie_id in enumerate(movie_ids)}

user_codes, user_ids = pd.factorize(ratings["userId"])

ratings = ratings[ratings["movieId"].isin(movie_to_index)].copy()
ratings["movie_index"] = ratings["movieId"].map(movie_to_index)

# Build a sparse movie-user ratings matrix.
rating_matrix = csr_matrix(
    (
        ratings["rating"].astype(float),
        (ratings["movie_index"], user_codes[ratings.index])
    ),
    shape=(len(movies), len(user_ids))
)

# Calculate the average rating and rating count.
stats = ratings.groupby("movieId")["rating"].agg(
    average_rating="mean",
    rating_count="count"
)

movie_stats = movies.merge(
    stats, on="movieId", how="left"
)

movie_stats["average_rating"] = (
    movie_stats["average_rating"].fillna(0).round(2)
)
movie_stats["rating_count"] = (
    movie_stats["rating_count"].fillna(0).astype(int)
)


def recommend_movies(selected_movie, number_of_recommendations):
    if not selected_movie:
        return pd.DataFrame(
            columns=["Movie", "Genres", "Average Rating", "Rating Count"]
        )

    match = movies[movies["title"] == selected_movie]

    if match.empty:
        return pd.DataFrame(
            columns=["Movie", "Genres", "Average Rating", "Rating Count"]
        )

    movie_id = int(match.iloc[0]["movieId"])
    movie_index = movie_to_index[movie_id]

    # Compare this movie's rating pattern with every other movie.
    similarities = cosine_similarity(
        rating_matrix[movie_index], rating_matrix
    ).flatten()

    similarities[movie_index] = -1

    # Prefer movies with sufficient ratings to make recommendations useful.
    popular_enough = (
        movie_stats["rating_count"].to_numpy() >= 5
    )
    similarities[~popular_enough] = -1

    top_indices = similarities.argsort()[::-1]
    top_indices = [
        i for i in top_indices
        if similarities[i] > 0
    ][:int(number_of_recommendations)]

    if not top_indices:
        return pd.DataFrame({
            "Message": [
                "No similar movies found. Try a more popular movie."
            ]
        })

    results = movie_stats.iloc[top_indices][
        ["title", "genres", "average_rating", "rating_count"]
    ].copy()

    results.columns = [
        "Movie", "Genres", "Average Rating", "Rating Count"
    ]

    return results.reset_index(drop=True)


movie_choices = sorted(movies["title"].dropna().unique().tolist())

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Movie Recommendation System"
) as demo:

    gr.Markdown("# Movie Recommendation System")
    gr.Markdown(
        "Discover movies using patterns in user ratings "
        "from the MovieLens dataset."
    )

    selected_movie = gr.Dropdown(
        choices=movie_choices,
        label="Choose a movie you like",
        value="Toy Story (1995)",
        filterable=True
    )

    number = gr.Slider(
        minimum=3,
        maximum=10,
        value=5,
        step=1,
        label="Number of recommendations"
    )

    recommend_button = gr.Button(
        "Recommend Movies", variant="primary"
    )

    output = gr.Dataframe(
        label="Recommended Movies",
        interactive=False
    )

    recommend_button.click(
        fn=recommend_movies,
        inputs=[selected_movie, number],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
