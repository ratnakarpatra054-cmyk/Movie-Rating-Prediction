"""
=============================================================
  Movie Rating Prediction — Inference Script
  Afame Technologies | Data Science Internship Project
  Author  : Ratnakar Patra
  Dataset : IMDb Indian Movies (movies_data.csv)
=============================================================

HOW TO USE
----------
  python src/predict.py

This script:
  1. Loads the saved model pipeline from  models/
  2. Uses the saved encoding maps for Director/Actor/Genre
  3. Shows predictions on 3 sample movies
  4. Lets you enter your own movie details interactively
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "movie_rating_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "encoding_maps.pkl")


def load_artifacts():
    """Load saved model pipeline and encoding maps."""
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Model not found. Please run:  python train_and_save.py")
        sys.exit(1)
    if not os.path.exists(META_PATH):
        print("[ERROR] Encoding maps not found. Please run:  python train_and_save.py")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    maps  = joblib.load(META_PATH)
    print("[OK] Model and encoding maps loaded.")
    return model, maps


def encode_input(raw: dict, maps: dict) -> pd.DataFrame:
    """
    Convert raw movie info dict into the numeric feature vector
    that the trained pipeline expects.

    Target Encoding replaces each category (Director, Actor, Genre)
    with the mean rating of all movies in that category from training data.
    If the category was never seen during training, we use the global mean.

    Parameters
    ----------
    raw  : dict with keys: Year, Duration, Genre, Director, Actor 1, Actor 2, Actor 3, Votes
    maps : dict loaded from encoding_maps.pkl

    Returns
    -------
    pd.DataFrame — one row, 9 numeric columns ready for prediction
    """
    gm = maps["global_mean"]

    # Extract primary genre
    primary_genre = str(raw.get("Genre", "Drama")).split(",")[0].strip()

    log_votes = np.log1p(float(raw.get("Votes", 1000)))
    year      = float(raw.get("Year", 2020))
    duration  = float(raw.get("Duration", 120))
    movie_age = 2024 - year

    genre_enc    = maps["genre_mean"].get(primary_genre,       gm)
    director_enc = maps["director_mean"].get(raw.get("Director", ""), gm)
    actor1_enc   = maps["actor1_mean"].get(raw.get("Actor 1", ""),   gm)
    actor2_enc   = maps["actor2_mean"].get(raw.get("Actor 2", ""),   gm)
    actor3_enc   = maps["actor3_mean"].get(raw.get("Actor 3", ""),   gm)

    return pd.DataFrame([{
        "Year"        : year,
        "Duration"    : duration,
        "Log_Votes"   : log_votes,
        "Movie_Age"   : movie_age,
        "Genre_enc"   : genre_enc,
        "Director_enc": director_enc,
        "Actor_1_enc" : actor1_enc,
        "Actor_2_enc" : actor2_enc,
        "Actor_3_enc" : actor3_enc,
    }])


def predict_rating(model, maps: dict, raw: dict) -> float:
    """Predict and clip rating to valid IMDb range [1.0, 10.0]."""
    X   = encode_input(raw, maps)
    raw_pred = model.predict(X)[0]
    return round(float(np.clip(raw_pred, 1.0, 10.0)), 2)


def interpret_rating(rating: float) -> str:
    """Return a human-friendly quality label for an IMDb rating."""
    if rating >= 9.0:   return "All-Time Classic   [*****]"
    elif rating >= 8.0: return "Excellent          [*****]"
    elif rating >= 7.0: return "Very Good          [****]"
    elif rating >= 6.0: return "Good               [***]"
    elif rating >= 5.0: return "Average            [**]"
    elif rating >= 4.0: return "Below Average      [*]"
    else:               return "Poor               [ ]"


def print_banner():
    print()
    print("=" * 62)
    print("    MOVIE RATING PREDICTOR  --  Afame Technologies")
    print("    IMDb Rating Prediction using Machine Learning")
    print("=" * 62)


def print_result(raw: dict, rating: float):
    label = interpret_rating(rating)
    print()
    print("  " + "-" * 56)
    print("  PREDICTION RESULT")
    print("  " + "-" * 56)
    print(f"  Year      : {int(raw.get('Year', 0))}")
    print(f"  Duration  : {int(raw.get('Duration', 0))} minutes")
    print(f"  Genre     : {raw.get('Genre', 'N/A')}")
    print(f"  Director  : {raw.get('Director', 'N/A')}")
    print(f"  Actor 1   : {raw.get('Actor 1', 'N/A')}")
    print(f"  Actor 2   : {raw.get('Actor 2', 'N/A')}")
    print(f"  Votes est.: {int(raw.get('Votes', 0)):,}")
    print("  " + "-" * 56)
    print(f"  Predicted IMDb Rating : {rating:.2f} / 10.0")
    print(f"  Quality Category      : {label}")
    print("  " + "-" * 56)
    print()


# ── Sample movies for demonstration ────────────────────────────
SAMPLE_MOVIES = [
    {
        "name"    : "Big-budget Tamil Action (2023)",
        "Year"    : 2023,
        "Duration": 150,
        "Genre"   : "Action, Drama",
        "Director": "S. Shankar",
        "Actor 1" : "Rajinikanth",
        "Actor 2" : "Akshay Kumar",
        "Actor 3" : "Amy Jackson",
        "Votes"   : 25000,
    },
    {
        "name"    : "Acclaimed Bollywood Drama (2019)",
        "Year"    : 2019,
        "Duration": 162,
        "Genre"   : "Comedy, Drama",
        "Director": "Rajkumar Hirani",
        "Actor 1" : "Aamir Khan",
        "Actor 2" : "Anushka Sharma",
        "Actor 3" : "Sanjay Dutt",
        "Votes"   : 300000,
    },
    {
        "name"    : "Low-budget Horror with Unknown Cast",
        "Year"    : 2018,
        "Duration": 92,
        "Genre"   : "Horror",
        "Director": "First Time Director",
        "Actor 1" : "Unknown Actor A",
        "Actor 2" : "Unknown Actor B",
        "Actor 3" : "Unknown Actor C",
        "Votes"   : 80,
    },
]


def main():
    print_banner()

    # Load model and encoding maps
    model, maps = load_artifacts()

    # Show model info
    info_path = os.path.join(BASE_DIR, "models", "model_info.txt")
    if os.path.exists(info_path):
        print()
        with open(info_path, "r") as f:
            print(f.read())

    # ── Sample Predictions ──────────────────────────────────────
    print("=" * 62)
    print("  SAMPLE PREDICTIONS (3 movies)")
    print("=" * 62)

    for movie in SAMPLE_MOVIES:
        name = movie.pop("name", "Movie")
        print(f"\n  [{name}]")
        rating = predict_rating(model, maps, movie)
        print_result(movie, rating)
        movie["name"] = name  # restore

    # ── Interactive Prediction ──────────────────────────────────
    print("=" * 62)
    print("  TRY YOUR OWN MOVIE")
    print("  (Press Enter to use the default value shown)")
    print("=" * 62)

    try:
        year     = input("\n  Release Year      [e.g. 2023] : ").strip()
        duration = input("  Duration (mins)   [e.g. 130]  : ").strip()
        genre    = input("  Genre             [e.g. Drama] : ").strip()
        director = input("  Director Name                  : ").strip()
        actor1   = input("  Lead Actor (Actor 1)           : ").strip()
        actor2   = input("  Actor 2                        : ").strip()
        actor3   = input("  Actor 3                        : ").strip()
        votes    = input("  Estimated Votes   [e.g. 5000]  : ").strip()

        custom = {
            "Year"    : int(year)     if year     else 2022,
            "Duration": int(duration) if duration else 120,
            "Genre"   : genre         if genre    else "Drama",
            "Director": director      if director else "Unknown Director",
            "Actor 1" : actor1        if actor1   else "Unknown Actor",
            "Actor 2" : actor2        if actor2   else "Unknown Actor",
            "Actor 3" : actor3        if actor3   else "Unknown Actor",
            "Votes"   : int(votes)    if votes    else 1000,
        }

        print("\n  [Your Custom Movie]")
        rating = predict_rating(model, maps, custom)
        print_result(custom, rating)

    except (KeyboardInterrupt, EOFError):
        print("\n\n  [Interactive mode skipped]\n")
    except ValueError as e:
        print(f"\n  [Input error: {e}] — please enter valid numbers for Year/Duration/Votes.\n")

    print("=" * 62)
    print("  Thank you! Project by Ratnakar Patra")
    print("  Afame Technologies | Data Science Internship")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
