"""
A movie recommender app built with streamlit.
"""

import numpy as np
import pandas as pd
import streamlit as st
import validators

from recommender import Recommender


@st.cache_data
def load_movies():
    """
    Function to load prepared data from CSV files.
    """
    # movies = pd.read_csv("./data/movies_prepared.csv")
    movies = pd.read_csv("./data/movies_imdb.csv")
    return movies


@st.cache_data
def get_random_movies_to_rate(num_movies=5):
    """
    Function to randomly get movie titles and ids to be rated by the user.
    """
    movies = load_movies()

    movies = movies.sort_values("imdb_rating", ascending=False).reset_index(drop=True)
    movies = movies[:100]

    select = np.random.choice(movies.index, size=num_movies, replace=False)

    return movies.iloc[select]


@st.cache_data
def get_movies():
    """
    Function to get movie titles and ids to be selected by the user.
    """
    movies = load_movies()
    movies = movies.sort_values("title").reset_index(drop=True)

    return movies


@st.cache_data
def get_movie_id_from_title(title_str):
    """
    Function that returns a movies ID from a title input.
    """
    movies = load_movies()
    movies = movies[movies["title"] == title_str]["movie_id"]

    return int(movies.iloc[0])


def prepare_query_favourites():
    """
    Function to prepare query to search for movies based on favourite movies.
    """
    data = get_movies()

    st.markdown(
        "Don't know which movie to watch tonight?"
        "Just **tell us some of your favourite movies** and based on that "
        "we'll recommend you something you might like."
    )

    user_ratings = st.multiselect(
        "Select as many movies as you like. Type to filter the list.",
        data["title"],
    )

    query = {}
    for title_selected in user_ratings:
        # Get movie ids
        mid = get_movie_id_from_title(title_selected)
        # Set rating to 5 for selected movies
        query[mid] = 5

    return query


def prepare_query_rating():
    """
    Function to prepare query to search for movies based on rating.
    """
    data = get_random_movies_to_rate(10)

    st.markdown(
        "Don't know which movie to watch tonight? Here are 10 randomly chosen movies."
        "Just **rate as many of them as you like** and based on your rating "
        "we'll recommend you something you might like."
    )

    query = {}
    for movie_id, title in zip(data["movie_id"], data["title"]):
        query[movie_id] = st.select_slider(title, options=[0, 1, 2, 3, 4, 5])

    return query


def recommender(rec_type="fav"):
    """
    Function to recommend movies.
    """

    # Prepare query based on type
    query = (
        prepare_query_rating() if rec_type == "rating" else prepare_query_favourites()
    )

    # Show select list for algorithm to use
    method_select = st.selectbox(
        "Select algorithm",
        ["Nearest Neighbors", "Non-negative matrix factorization"],
        key="method_selector_" + rec_type,
    )

    # Translate selection into keywords
    method = "neighbors" if method_select == "Nearest Neighbors" else "nmf"

    num_movies = st.slider(
        "How many movies should we recommend?",
        min_value=1,
        max_value=10,
        value=5,
        key="num_movies_slider_" + rec_type,
    )

    # Start recommender
    if st.button("Recommend some movies!", key="button_" + rec_type):
        with st.spinner(f"Calculating recommendations using {method_select}..."):
            recommend = Recommender(query, method=method, k=num_movies)
            movie_ids, _ = recommend.recommend()

        with st.spinner("Fetching movie information from IMDB..."):
            st.write("Recommended movies using Nearest Neighbors:\n")
            for movie_id in movie_ids:
                display_movie(movie_id)


def display_movie(movie_id):
    """
    Function that displays a movie with information from IMDB.
    """
    movies = load_movies()
    movie = movies[movies["movie_id"] == movie_id].copy()

    col1, col2 = st.columns([1, 4])

    with col1:
        if validators.url(str(movie["cover_url"].iloc[0])):
            st.image(movie["cover_url"].iloc[0])

    with col2:
        if "title" in movie.columns and "year" in movie.columns:
            st.header(f"{movie['title'].iloc[0]} ({movie['year'].iloc[0]})")
        if "imdb_rating" in movie.columns:
            st.markdown(f"**IMDB-rating:** {movie['imdb_rating'].iloc[0]}/10")
        if "genre" in movie.columns:
            st.markdown(f"**Genres:** {', '.join(movie['genre'].iloc[0].split(' | '))}")
        if "director" in movie.columns:
            st.markdown(
                f"**Director(s):** {', '.join(movie['director'].iloc[0].split('|'))}"
            )
        if "cast" in movie.columns:
            st.markdown(
                f"**Cast:** {', '.join(movie['cast'].iloc[0].split('|')[:10])}, ..."
            )
        if "plot" in movie.columns:
            st.markdown(f"{movie['plot'].iloc[0]}")
        if validators.url(str(movie["url"].iloc[0])):
            st.markdown(f"[Read more on imdb.com]({movie['url'].iloc[0]})")
    st.divider()


# Set page title
st.set_page_config(page_title="What should I watch tonight? | Your movie recommender")

# Header image
st.image("data/cover_collage.jpg")

# Print title and subtitle
st.title("What should I watch tonight?")
st.subheader("Your personal movie recommender")

tab1, tab2 = st.tabs(["By favourite movies", "By rating"])

with tab1:
    recommender("fav")

with tab2:
    recommender("rating")
