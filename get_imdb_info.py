"""
Fetch movie information from IMDB
"""

import time

import pandas as pd
from imdb import Cinemagoer
from imdb.helpers import resizeImage


def load_movies():
    """
    Function to load prepared data from CSV files.
    """
    movies = pd.read_csv("./data/movies_prepared.csv")
    return movies


def fetch_from_imdb(movie_title):
    """
    Function to fetch information from IMDB
    """
    imdb = Cinemagoer()
    imdb_movies = imdb.search_movie(movie_title)
    time.sleep(0.5)

    if len(imdb_movies) > 0:
        imdb_movie = imdb.get_movie(imdb_movies[0].movieID)

        # Define keys to be included
        keys = {
            "imdbID": "imdb_id",
            "title": "title",
            "year": "year",
            "rating": "imdb_rating",
            "genre": "genre",
            "director": "director",
            "cast": "cast",
            "full-size cover url": "cover_url",
            "plot outline": "plot",
        }

        metadata = {}
        for key, value in keys.items():
            try:
                if key in ["director", "cast"]:
                    metadata[value] = [content["name"] for content in imdb_movie[key]]
                elif key == "full-size cover url":
                    metadata[value] = resizeImage(imdb_movie[key], width=200)
                else:
                    metadata[value] = imdb_movie[key]
            except KeyError:
                print(f"KeyError: {key} doesn't exist for {imdb_movie['title']}.")

        print(f"Fetched metadata for {imdb_movie['title']}.")
        return metadata

    print(f"No movie info returned for {movie_title}.")
    return False


def update_dataframe_row(df_row):
    """
    Function to update a dataframe row with IMDB information.
    """
    metadata = fetch_from_imdb(df_row["title"])

    if isinstance(metadata, dict):
        for key, value in metadata.items():
            df_row[key] = value

    return df_row


def main():
    """
    Main function
    """
    movies = load_movies()

    # Limit number of movies for testing
    imdb_movies = movies.iloc[:100]
    imdb_movies = imdb_movies.apply(update_dataframe_row, axis=1)

    file_name = "data/data_with_imdb.csv"
    imdb_movies[
        [
            "movie_id",
            "imdb_id",
            "title",
            "imdb_rating",
            "year",
            "genre",
            "director",
            "cast",
            "cover_url",
            "plot",
        ]
    ].to_csv(file_name)

    print(f"File saved to {file_name}.")


if __name__ == "__main__":
    main()
