"""

Write a program that checks that recommenders works as expected

We will use pytest
to install pytest run in the terminal 
+ pip install pytest
/ conda install pytest

TDD (Test-driven-development)  cycle:

0. Make an Hypothesis:
    the units/programs work
1. Write test that fails (to disprove the Hypothesis)
2. Change the code so that the Hypothesis is re-established
3. repeat 0-->2

"""
from recommender import Recommender


def test_for_5_movies_neighbors():
    """
    Test if Recommender returns the correct number of movies.
    """
    user_query = {10: 4, 100: 3, 555: 3.5, 756: 2, 1224: 5}

    recommend = Recommender(user_query, method="neighbors", k=5)
    _, titles = recommend.recommend()

    assert len(titles) == 5


def test_for_5_movies_nmf():
    """
    Test if Recommender returns the correct number of movies.
    """
    user_query = {10: 4, 100: 3, 555: 3.5, 756: 2, 1224: 5}

    recommend = Recommender(user_query, method="nmf", k=5)
    _, titles = recommend.recommend()

    assert len(titles) == 5
