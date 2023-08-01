![Cover collage](data/cover_collage.jpg)

# Movie recommender

### A Streamlit app to recommend movies based on user input

Uses two different models to determine the recommendations: **Nearest Neighbors** and **Non-negative Matrix Factorization** (can be selected). The models are trained on a reduced data set of movie ratings.

There are two methods available to get reccomendations:

- **By favourite movies:** Select as many movies you like and get a recommendation for similar movies.
- **By rating:** Rate up to 10 arbitraily selected movies and get a recommendation based on your rating.

This App was a weekly project at the SPICED Datascience Bootcamp from April to June 2023.

### Try the App

To use the Streamlit app, clone the repository, install the requirements to your Python environment and run the app:

```bash
git clone https://github.com/yotkadata/movie_recommender
cd movie_recommender/
pip install -r requirements.txt
streamlit run app.py
```

This should open a page in your default browser at http://localhost:8501 that shows the app.

### Screenshots

<p float="left">
  <a href="https://github.com/yotkadata/movie_recommender/blob/main/data/screenshots/frontpage.png">
    <img src="https://github.com/yotkadata/movie_recommender/blob/main/data/screenshots/frontpage.png?raw=true" width="250" />
  </a>
  <a href="https://github.com/yotkadata/movie_recommender/blob/main/data/screenshots/recommendation-by-favorite-movies.png">
    <img src="https://github.com/yotkadata/movie_recommender/blob/main/data/screenshots/recommendation-by-favorite-movies.png?raw=true" width="250" />
  </a>
  <a href="https://github.com/yotkadata/movie_recommender/blob/main/data/screenshots/method-by-rating.png">
    <img src="https://github.com/yotkadata/movie_recommender/blob/main/data/screenshots/method-by-rating.png?raw=true" width="250" />
  </a>
</p>
