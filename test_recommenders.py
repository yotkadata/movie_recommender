"""

Unit tests (pytest) for the Recommender class.
"""
import pandas as pd
import pytest

from recommender import Recommender


@pytest.fixture(name="recommender_nmf_instance")
def fixture_recommender_nmf_instance():
    """
    Fixture to create an instance of Recommender with NMF method for testing.
    """
    query = {10: 4, 100: 3, 555: 3.5, 756: 2, 1224: 5}
    return Recommender(query, method="nmf", k=5)


@pytest.fixture(name="recommender_neighbors_instance")
def fixture_recommender_neighbors_instance():
    """
    Fixture to create an instance of Recommender with Neighbors method for testing.
    """
    query = {10: 4, 100: 3, 555: 3.5, 756: 2, 1224: 5}
    return Recommender(query, method="neighbors", k=5)


def test_recommender_nmf(recommender_nmf_instance):
    """
    Test the "recommender_nmf" method to ensure it
    recommends movies correctly using NMF method.
    """
    movie_ids, titles = recommender_nmf_instance.recommend()

    # Assert that the output is a list of k movie ids.
    assert isinstance(
        movie_ids, pd.Series
    ), f"movie_ids should be a Pandas Series, {type(movie_ids)} found."
    assert (
        len(movie_ids) == 5
    ), f"movie_ids should be a Pandas Series of 5 movie ids, {len(movie_ids)} found."

    # Assert that the output titles are also a list of k titles.
    assert isinstance(titles, list), f"titles should be a list, {type(titles)} found."
    assert (
        len(titles) == 5
    ), f"titles should be a list of 5 titles, {len(titles)} found."


def test_recommender_neighbors(recommender_neighbors_instance):
    """
    Test the "recommender_neighbors" method to ensure it
    recommends movies correctly using Neighbors method.
    """
    movie_ids, titles = recommender_neighbors_instance.recommend()

    # Assert that the output is a list of k movie ids.
    assert isinstance(
        movie_ids, pd.Series
    ), f"movie_ids should be a Pandas Series, {type(movie_ids)} found."
    assert (
        len(movie_ids) == 5
    ), f"movie_ids should be a Pandas Series of 5 movie ids, {len(movie_ids)} found."

    # Assert that the output titles are also a list of k titles.
    assert isinstance(titles, list), f"titles should be a list, {type(titles)} found."
    assert (
        len(titles) == 5
    ), f"titles should be a list of 5 titles, {len(titles)} found."


def test_recommender_invalid_method():
    """
    Test if an invalid method is provided for the
    Recommender class and expect a ValueError.
    """
    query = {10: 4, 100: 3, 555: 3.5, 756: 2, 1224: 5}
    with pytest.raises(ValueError):
        Recommender(query, method="invalid_method", k=5)


def test_get_movie_ids():
    """
    Test the "get_movie_ids" method to ensure it
    returns a Pandas Series.
    """
    recommender = Recommender({})
    movie_ids = recommender.get_movie_ids()
    assert isinstance(
        movie_ids, pd.Series
    ), f"movie_ids should be a Pandas Series, {type(movie_ids)} found."


def test_get_movie_titles_by_ids():
    """
    Test the "get_movie_titles_by_ids" method to ensure
    it returns a list of movie titles.
    """
    movie_ids = [1, 2, 3]
    recommender = Recommender({})
    titles = recommender.get_movie_titles_by_ids(movie_ids)
    assert isinstance(titles, list), f"titles should be a list, {type(titles)} found."
    assert len(titles) == len(
        movie_ids
    ), f"titles should be a list of 5 titles, {len(titles)} found."


def test_load_model():
    """
    Test the "load_model" method to ensure it loads a model
    from a pickle file successfully.
    """
    recommender = Recommender({})
    model = recommender.load_model("data/model_nmf.pkl")
    assert model is not None


def test_load_prepared_data():
    """
    Test the "load_prepared_data" method to ensure it loads
    prepared data from CSV files correctly.
    """
    recommender = Recommender({})
    movies, ratings = recommender.load_prepared_data()
    assert isinstance(
        movies, pd.DataFrame
    ), f"movies should be a Pandas Dataframe, {type(movies)} found."
    assert isinstance(
        ratings, pd.DataFrame
    ), f"ratings should be a Pandas Dataframe, {type(ratings)} found."


if __name__ == "__main__":
    pytest.main()
