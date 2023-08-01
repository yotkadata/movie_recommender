"""
Build and save models for the recommender
"""

import pickle

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors


def build_model_nmf(n_components: int = 2000, max_iter: int = 1000) -> str:
    """
    Function to build and save a recommender model using NMF.
    """
    # Load prepared data
    ratings = pd.read_csv("data/ratings_prepared.csv")

    # Initialize a sparse user-item rating matrix
    r_matrix = csr_matrix(
        (ratings["rating"], (ratings["user_id"], ratings["movie_id"]))
    )

    # Instantiate model and fit
    model = NMF(n_components=n_components, max_iter=max_iter)
    print(
        "NMF model instantiated with following hyperparameters:\n"
        f"n_components={n_components}\n"
        f"max_iter={max_iter}\n"
        "Starting to fit.\n"
    )

    # Fit it to the Ratings matrix
    model.fit(r_matrix)

    # Print reconstruction error
    print(f"NMF model built. Reconstruction error: {model.reconstruction_err_}")

    # Save model
    file_name = "data/model_nmf.pkl"
    with open(file_name, "wb") as file:
        pickle.dump(model, file)

    return file_name


def build_model_neighbors(metric: str = "cosine", n_jobs: int = -1) -> str:
    """
    Function to build and save a recommender model using Nearest Neighbors.
    """
    # Load prepared data
    ratings = pd.read_csv("data/ratings_prepared.csv")

    # Initialize a sparse user-item rating matrix
    r_matrix = csr_matrix(
        (ratings["rating"], (ratings["user_id"], ratings["movie_id"]))
    )

    # Initialize the NearestNeighbors model
    model = NearestNeighbors(metric=metric, n_jobs=n_jobs)
    print(
        "Nearest Neighbors model instantiated with following hyperparameters:\n"
        f"metric={metric}\n"
        f"n_jobs={n_jobs}\n\n"
        "Starting to fit.\n"
    )

    # Fit it to the Ratings matrix
    model.fit(r_matrix)

    # Print reconstruction error
    print("Nearest neighbor model built.")

    # Save model
    file_name = "data/model_neighbors.pkl"
    with open(file_name, "wb") as file:
        pickle.dump(model, file)

    return file_name


def main() -> None:
    """
    Main function
    """

    file_name_nmf = build_model_nmf()
    print(f"NMF model saved to {file_name_nmf}.")

    file_name_neighbors = build_model_neighbors()
    print(f"Nearest Neighbors model saved to {file_name_neighbors}.")


if __name__ == "__main__":
    main()
