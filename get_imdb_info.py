"""
Fetch movie information from IMDB
"""

import time
from pathlib import Path

import pandas as pd
from imdb import Cinemagoer
from imdb.helpers import resizeImage


def load_movies() -> pd.DataFrame:
    """
    Function to load prepared data from CSV files.
    """
    movies = pd.read_csv("./data/movies_prepared.csv")
    return movies


def fetch_from_imdb(movie_title: str) -> dict:
    """
    Function to fetch information from IMDB
    """

    # Fix some missing values manually
    if movie_title == "Seven (a.k.a. Se7en) (1995)":
        movie_title = "Se7en"

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
                    metadata[value] = "|".join(
                        [content["name"] for content in imdb_movie[key]]
                    )
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


def update_dataframe_row(df_row: pd.Series) -> pd.Series:
    """
    Function to update a dataframe row with IMDB information.
    """
    metadata = fetch_from_imdb(df_row["title"])

    if isinstance(metadata, dict):
        for key, value in metadata.items():
            df_row[key] = value

    return df_row


def combine_batches() -> str:
    """
    Function to combine CSV batches.
    """
    # Get all CSV files from batch directory
    batch_dir = Path("data/batches/")
    csv_files = list(batch_dir.glob("*.csv"))

    df_csv_concat = pd.concat(
        [pd.read_csv(file, index_col=[0]) for file in csv_files], ignore_index=True
    )

    concat_file = "data/data_with_imdb.csv"
    df_csv_concat.to_csv(concat_file)
    print(f"Batches combined to {concat_file}.")

    # Remove batch files
    for file in csv_files:
        file.unlink()

    print("Batch files removed.")

    return concat_file


def main() -> None:
    """
    Main function
    """
    movies = load_movies()

    batch_size = 50
    total_num = len(movies)

    splits = list(range(0, total_num, batch_size))
    splits.append(total_num + 1)

    batch_list = [(splits[i], splits[i + 1]) for i in range(len(splits) - 1)]

    for batch in batch_list:
        file_name = Path(f"data/batches/data_with_imdb_{batch[0]}-{batch[1]-1}.csv")

        if not file_name.is_file():
            df_new = movies.iloc[batch[0] : batch[1]].apply(
                update_dataframe_row, axis=1
            )

            cols = [
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
            df_new[cols].to_csv(file_name)

            print(f"Batch saved to {file_name}.")
        else:
            print(f"Batch {file_name.name} already exists, skipping.")

    combine_batches()


if __name__ == "__main__":
    main()
