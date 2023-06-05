"""
A movie recommender app built with streamlit.
"""

import numpy as np
import pandas as pd
import streamlit as st
from imdb import Cinemagoer
from imdb.helpers import resizeImage

from recommender import Recommender


@st.cache_data
def load_movies():
    """
    Function to load prepared data from CSV files.
    """
    movies = pd.read_csv("./data/movies_prepared.csv")
    return movies


@st.cache_data
def get_random_movies_to_rate(num_movies=5):
    """
    Function to randomly get movie titles and ids to be rated by the user.
    """
    movies = load_movies()

    movies = movies.sort_values("rating", ascending=False).reset_index(drop=True)
    movies = movies[:100]

    select = np.random.choice(movies.index, size=num_movies, replace=False)

    return movies.iloc[select]


@st.cache_data
def get_movies(num_movies=5):
    """
    Function to get movie titles and ids to be selected by the user.
    """
    movies = load_movies()
    if num_movies == "all":
        num_movies = len(movies)

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
    data = get_movies("all")

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
        "Just **rate as many of them as you like** and based on your rating we'll recommend you something you might like."
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

    # Translate selection into keaywords
    method = "neighbors" if method_select == "Nearest Neighbors" else "nmf"

    num_movies = st.slider(
        "How many movies should we recommend?",
        min_value=1,
        max_value=10,
        value=3,
        key="num_movies_slider_" + rec_type,
    )

    # Start recommender
    if st.button("Recommend some movies!", key="button_" + rec_type):
        with st.spinner(f"Calculating recommendations using {method_select}..."):
            recommend = Recommender(query, method=method, k=num_movies)
            _, titles = recommend.recommend()

        with st.spinner("Fetching movie information from IMDB..."):
            st.write("Recommended movies using Nearest Neighbors:\n")
            for title in titles:
                imdb = Cinemagoer()
                imdb_movies = imdb.search_movie(title)
                imdb_movie = imdb.get_movie(imdb_movies[0].movieID)
                display_movie(imdb_movie)


def display_movie(movie):
    """
    Function that displays a movie with information from IMDB.
    """
    directors = [director["name"] for director in movie["director"]]
    cast = [actor["name"] for actor in movie["cast"]]
    img_url = resizeImage(movie["full-size cover url"], width=200)

    col1, col2 = st.columns([1, 4])

    with col1:
        st.image(img_url)

    with col2:
        st.header(f"{movie['title']} ({movie['year']})")
        st.markdown(f"**IMDB-rating:** {movie['rating']}/10")
        st.markdown(f"**Genres:** {', '.join(movie['genres'])}")
        st.markdown(f"**Director(s):** {', '.join(directors)}")
        st.markdown(f"**Cast:** {', '.join(cast[:10])}, ...")
        st.markdown(f"{movie['plot outline']}")
    st.divider()


# Set page title
st.set_page_config(page_title="What should I watch tonight? | Your movie recommender")

# Print title and subtitle
st.title("What should I watch tonight?")
st.subheader("Your personal movie recommender")

tab1, tab2 = st.tabs(["By favourite movies", "By rating"])

with tab1:
    recommender("fav")

with tab2:
    recommender("rating")
