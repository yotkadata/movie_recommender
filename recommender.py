"""
Class to recommend movies.
"""

import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class Recommender:
    """
    Class to recommend movies based on user ratings input.
    """

    def __init__(self, query, method="neighbors", k=10) -> None:
        self.query = query
        self.method = method
        self.k = k

    def recommend(self):
        """
        Recommends the top k movies for any given input query.
        Returns a list of k movie ids and corresponding movie titles.
        """

        # Create user vector
        movie_ids = self.get_movie_ids()
        df_query = pd.DataFrame(self.query, columns=movie_ids, index=[0])

        # Fill missing values
        df_query_filled = df_query.fillna(0)

        if self.method == "nmf":
            # Use Non-negative Matrix Factorization (NMF)
            movie_ids = self.recommender_nmf(df_query_filled)
        else:
            # Use Nearest Neighbors
            movie_ids = self.recommender_neighbors(df_query_filled)

        # Get corresponding titles in the same order
        titles = self.get_movie_titles_by_ids(movie_ids)

        return movie_ids, titles

    def recommender_nmf(self, df_query):
        """
        Filters and recommends the top k movies for any given input query
        based on a trained NMF model.
        Returns a list of k movie ids.
        """
        # Load the model from file
        model = self.load_model("data/model_nmf.pkl")

        # Create user-feature matrix P for new user
        p_matrix = model.transform(df_query)

        # Reconstruct the user-movie(item) matrix/dataframe for the new user
        q_matrix = model.components_

        r_hat_matrix = np.dot(p_matrix, q_matrix)
        df_r_hat = pd.DataFrame(r_hat_matrix, columns=self.get_movie_ids())

        # Get a list of k top rated movies to recommend to the new user
        ranked = df_r_hat.T.sort_values(0, ascending=False)
        recommended = ranked[~ranked.index.isin(self.query)].reset_index()

        # Get movie ids of k best rated movies
        movie_ids = recommended.iloc[: self.k, 0]

        return movie_ids

    def recommender_neighbors(self, df_query):
        """
        Filters and recommends the top k movies for any given input query
        based on a trained nearest neighbors model.
        Returns a list of k movie ids.
        """
        # Load the model from file
        model = self.load_model("data/model_neighbors.pkl")

        # Calculate the distances to other users
        similarity_scores, neighbor_ids = model.kneighbors(
            df_query, n_neighbors=5, return_distance=True
        )

        # Save ids and scores in a DataFrame and sort it
        df_neighbors = pd.DataFrame(
            data={
                "neighbor_id": neighbor_ids[0],
                "similarity_score": similarity_scores[0],
            }
        )
        df_neighbors.sort_values("similarity_score", ascending=False, inplace=True)

        # Load ratings
        _, ratings = self.load_prepared_data()

        # Calculate CSR Matrix (R) and convert do Dataframe
        r_matrix = csr_matrix(
            (ratings["rating"], (ratings["user_id"], ratings["movie_id"]))
        )
        df_r = pd.DataFrame(r_matrix.todense())

        # Filter to only show similar users and filter out movies rated by the user
        neighborhood_filtered = df_r.iloc[neighbor_ids[0]].drop(
            self.query.keys(), axis=1
        )

        # Multiply the ratings with the similarity score of each user and
        # calculate the summed up rating for each movie
        df_score = neighborhood_filtered.apply(
            lambda x: df_neighbors.set_index("neighbor_id").loc[x.index][
                "similarity_score"
            ]
            * x
        )
        df_score_ranked = (
            df_score.sum(axis=0).reset_index().sort_values(0, ascending=False)
        )
        df_score_ranked.reset_index(drop=True, inplace=True)

        # Get movie ids of k best rated movies
        movie_ids = df_score_ranked.iloc[: self.k, 0]

        return movie_ids

    def load_model(self, file_name):
        """
        Function to load a model from a pickle file.
        """
        with open(file_name, "rb") as file:
            model = pickle.load(file)

        return model

    def load_prepared_data(self):
        """
        Function to load prepared data from CSV files.
        """
        ratings = pd.read_csv("./data/ratings_prepared.csv")
        movies = pd.read_csv("./data/movies_prepared.csv")
        return movies, ratings

    def get_movie_ids(self):
        """
        Function to get movie ids.
        """
        movies, _ = self.load_prepared_data()
        return movies["movie_id"]

    def get_movie_titles_by_ids(self, movie_ids):
        """
        Function to get movie ids.
        """
        movies, _ = self.load_prepared_data()
        titles = movies[movies["movie_id"].isin(movie_ids)]["title"].tolist()

        return titles


def main():
    """
    Main function.
    """
    # movie_id: rating
    user_query = {
        10: 4,  # Billy Madison (1995)
        100: 3,  # Bambi (1942)
        555: 3.5,  # Mortal Kombat (1995)
        756: 2,  # Inside Man (2006)
        1224: 5,  # Babe: Pig in the City (1998)
    }

    # Recommended movies using Nearest Neighbors
    recommend = Recommender(user_query, method="neighbors", k=5)
    _, titles = recommend.recommend()

    print("Recommended movies using Nearest Neighbors:\n")
    for i, title in enumerate(titles):
        print(f"{i+1}. {title}")

    print("")

    # Recommended movies using NMF
    recommend = Recommender(user_query, method="nmf", k=5)
    _, titles = recommend.recommend()

    print("Recommended movies using NMF:\n")
    for i, title in enumerate(titles):
        print(f"{i+1}. {title}")


if __name__ == "__main__":
    main()
