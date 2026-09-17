"""
Convert raw MovieLens-100K files into the exact input format expected by the
QIRS knowledge-graph builders (src/graph.py / src/graph_text.py), i.e. the
users_file / ratings_file / movie_mapping_file paths configured in
config.yaml.

If --ml-100k-dir doesn't exist yet (or is missing u.data/u.item/u.user), this
script downloads and extracts MovieLens-100K there automatically from
https://files.grouplens.org/datasets/movielens/ml-100k.zip. You can also do
this yourself first if you'd rather not rely on the automatic download:
    curl -o ml-100k.zip https://files.grouplens.org/datasets/movielens/ml-100k.zip
    unzip ml-100k.zip

Usage:
    python src/prepare_movielens.py --ml-100k-dir ml-100k

This writes, by default, exactly the three files config.yaml already points
at (relative to the repo root):
    data/Input/users.csv
    data/Input/ratings_array.json
    data/Input/movie_id_mapping.json

and, as a convenience, a fourth file (not read by the pipeline itself):
    data/Input/movie_titles.json
a plain list of every movie title, meant to be pasted into config.yaml's
`movie_list` (or a slice of it, via --max-movies) so the KG builder knows
which movies to enrich via the OpenAI API.

Schema notes (reverse-engineered from src/graph_text.py):
- users.csv columns: _id, gender, job, state, dob, languages
    - gender must be "Male"/"Female"/other (matched case-insensitively
      downstream); MovieLens only has M/F, mapped accordingly.
    - dob must parse with datetime.strptime(dob, "%d-%m-%Y"); MovieLens only
      gives age, so a synthetic Jan-1 birth date is derived from it.
    - languages must be a JSON-encoded list string, e.g. '["English"]'.
    - state has no MovieLens equivalent and is left blank.
- ratings_array.json: a list of {"_id": <user_id>, "rated": {<movie_id>:
  [<rating>], ...}}, where <user_id> must match a users.csv `_id` and
  <rating> must be the string "-1", "0", or "1" (not a 1-5 star score).
  MovieLens' 1-5 star ratings are mapped: >=4 -> "1", ==3 -> "0", <=2 -> "-1".
- movie_id_mapping.json: {<movie_id>: <title>}, movie_id matching the keys
  used in ratings_array.json's "rated" dict. MovieLens titles like
  "Toy Story (1995)" have the trailing "(YYYY)" stripped.
"""

import argparse
import json
import os
import shutil
import urllib.request
import zipfile
from collections import defaultdict

DEFAULT_ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
REQUIRED_FILES = ["u.data", "u.item", "u.user"]


def ensure_ml100k(ml_100k_dir: str, url: str = DEFAULT_ML100K_URL) -> str:
    """
    Make sure ml_100k_dir contains u.data/u.item/u.user, downloading and
    extracting MovieLens-100K there automatically if it doesn't. Returns
    ml_100k_dir unchanged either way (this never picks a different path).
    """
    if all(os.path.exists(os.path.join(ml_100k_dir, f)) for f in REQUIRED_FILES):
        return ml_100k_dir

    print(f"'{ml_100k_dir}' not found or incomplete - downloading MovieLens-100K from {url} ...")

    parent_dir = os.path.dirname(os.path.abspath(ml_100k_dir)) or "."
    os.makedirs(parent_dir, exist_ok=True)
    zip_path = os.path.join(parent_dir, "ml-100k.zip")

    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded to {zip_path}, extracting ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(parent_dir)
    except Exception as e:
        raise SystemExit(
            f"Automatic download of MovieLens-100K failed: {e}\n"
            f"Download and unzip it yourself instead, then pass the extracted folder:\n"
            f"  curl -o ml-100k.zip {url}\n"
            f"  unzip ml-100k.zip\n"
            f"  python src/prepare_movielens.py --ml-100k-dir ml-100k ..."
        )
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)

    # The zip's internal layout extracts to "<parent_dir>/ml-100k" regardless
    # of what ml_100k_dir was actually named - move it into place if they differ.
    extracted_dir = os.path.join(parent_dir, "ml-100k")
    if os.path.abspath(extracted_dir) != os.path.abspath(ml_100k_dir):
        if os.path.exists(ml_100k_dir):
            shutil.rmtree(ml_100k_dir)
        shutil.move(extracted_dir, ml_100k_dir)

    if not all(os.path.exists(os.path.join(ml_100k_dir, f)) for f in REQUIRED_FILES):
        raise SystemExit(
            f"Downloaded and extracted MovieLens-100K, but {REQUIRED_FILES} are "
            f"still missing from {ml_100k_dir} - the zip's internal layout may "
            f"have changed. Check {ml_100k_dir} manually."
        )

    print(f"MovieLens-100K ready at {ml_100k_dir}")
    return ml_100k_dir


def load_movies(ml_100k_dir: str) -> dict:
    """Returns {movie_id (str): title (str, year suffix stripped)}."""
    movies = {}
    path = os.path.join(ml_100k_dir, "u.item")
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 2:
                continue
            movie_id, raw_title = parts[0], parts[1]
            # Strip a trailing " (YYYY)" release-year suffix, e.g.
            # "Toy Story (1995)" -> "Toy Story"
            title = raw_title
            if title.endswith(")") and "(" in title:
                paren_start = title.rfind("(")
                candidate_year = title[paren_start + 1:-1]
                if candidate_year.isdigit():
                    title = title[:paren_start].strip()
            movies[movie_id] = title
    return movies


def load_users(ml_100k_dir: str) -> dict:
    """Returns {user_id (str): {"age": int, "gender": str, "occupation": str}}."""
    users = {}
    path = os.path.join(ml_100k_dir, "u.user")
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 4:
                continue
            user_id, age, gender, occupation = parts[0], parts[1], parts[2], parts[3]
            users[user_id] = {
                "age": int(age),
                "gender": gender,
                "occupation": occupation,
            }
    return users


def load_ratings(ml_100k_dir: str) -> list:
    """Returns a list of (user_id, movie_id, rating_int) tuples."""
    ratings = []
    path = os.path.join(ml_100k_dir, "u.data")
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            user_id, movie_id, rating = parts[0], parts[1], int(parts[2])
            ratings.append((user_id, movie_id, rating))
    return ratings


def to_ternary_rating(stars: int) -> str:
    """Map MovieLens' 1-5 star rating to the "-1"/"0"/"1" scheme graph_text.py expects."""
    if stars >= 4:
        return "1"
    if stars == 3:
        return "0"
    return "-1"


def to_gender_label(code: str) -> str:
    return {"M": "Male", "F": "Female"}.get(code, "Other")


def synthetic_dob(age: int, as_of_year: int = 1998) -> str:
    """MovieLens-100K has no DOB, only age (as of ~1997-1998 data collection).
    Produce a "%d-%m-%Y" string the users.csv loader can parse."""
    birth_year = max(1900, as_of_year - age)
    return f"01-01-{birth_year}"


def write_users_csv(users: dict, out_path: str) -> None:
    import csv

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["_id", "gender", "job", "state", "dob", "languages"])
        for user_id, u in users.items():
            writer.writerow([
                user_id,
                to_gender_label(u["gender"]),
                u["occupation"],
                # No state equivalent in MovieLens-100K. Deliberately not left
                # blank or "N/A": pandas.read_csv() treats both an empty field
                # and the literal string "N/A" as NaN (a float) by default,
                # and bool(float("nan")) is True in Python, which would make
                # downstream "if state:" checks treat it as a real value and
                # print things like "State: nan."
                "Unknown",
                synthetic_dob(u["age"]),
                json.dumps(["English"]),
            ])


def write_ratings_json(ratings: list, out_path: str) -> None:
    by_user = defaultdict(dict)
    for user_id, movie_id, stars in ratings:
        by_user[user_id][movie_id] = [to_ternary_rating(stars)]

    records = [{"_id": user_id, "rated": rated} for user_id, rated in by_user.items()]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def write_movie_mapping_json(movies: dict, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(movies, f, indent=2, ensure_ascii=False)


def write_movie_titles_json(movies: dict, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sorted(set(movies.values())), f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw MovieLens-100K into QIRS's expected users_file/ratings_file/movie_mapping_file format"
    )
    parser.add_argument("--ml-100k-dir", type=str, default="ml-100k",
                         help="Path to the extracted ml-100k folder (containing u.data, u.item, "
                              "u.user). Downloaded and extracted here automatically if missing (default: ml-100k)")
    parser.add_argument("--ml-100k-url", type=str, default=DEFAULT_ML100K_URL,
                         help="URL to download MovieLens-100K from if --ml-100k-dir is missing/incomplete")
    parser.add_argument("--output-dir", type=str, default="data/Input",
                         help="Where to write users.csv / ratings_array.json / movie_id_mapping.json (default: data/Input, matching config.yaml)")
    parser.add_argument("--max-users", type=int, default=None,
                         help="Optional cap on number of users to include (for a quick/cheap test run)")
    parser.add_argument("--max-movies", type=int, default=None,
                         help="Optional cap on number of movies to include (for a quick/cheap test run); "
                              "ratings for excluded movies are dropped")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ml_100k_dir = ensure_ml100k(args.ml_100k_dir, args.ml_100k_url)

    print(f"Reading MovieLens-100K from {ml_100k_dir} ...")
    movies = load_movies(ml_100k_dir)
    users = load_users(ml_100k_dir)
    ratings = load_ratings(ml_100k_dir)
    print(f"Loaded {len(movies)} movies, {len(users)} users, {len(ratings)} ratings")

    if args.max_movies is not None:
        keep_movie_ids = set(list(movies.keys())[:args.max_movies])
        movies = {mid: t for mid, t in movies.items() if mid in keep_movie_ids}
        ratings = [r for r in ratings if r[1] in keep_movie_ids]
        print(f"Capped to {len(movies)} movies -> {len(ratings)} ratings remain")

    if args.max_users is not None:
        keep_user_ids = set(list(users.keys())[:args.max_users])
        users = {uid: u for uid, u in users.items() if uid in keep_user_ids}
        ratings = [r for r in ratings if r[0] in keep_user_ids]
        print(f"Capped to {len(users)} users -> {len(ratings)} ratings remain")

    users_path = os.path.join(args.output_dir, "users.csv")
    ratings_path = os.path.join(args.output_dir, "ratings_array.json")
    mapping_path = os.path.join(args.output_dir, "movie_id_mapping.json")
    titles_path = os.path.join(args.output_dir, "movie_titles.json")

    write_users_csv(users, users_path)
    write_ratings_json(ratings, ratings_path)
    write_movie_mapping_json(movies, mapping_path)
    write_movie_titles_json(movies, titles_path)

    print(f"Wrote {users_path}")
    print(f"Wrote {ratings_path}")
    print(f"Wrote {mapping_path}")
    print(f"Wrote {titles_path} ({len(set(movies.values()))} unique titles)")
    print()
    print("Next steps:")
    print("  1. In config.yaml, set movie_list to some/all titles from movie_titles.json")
    print("     (each title triggers OpenAI API calls to enrich it, so start with a small slice).")
    print("  2. Confirm users_file/ratings_file/movie_mapping_file in config.yaml point at:")
    print(f"       {users_path}")
    print(f"       {ratings_path}")
    print(f"       {mapping_path}")
    print("  3. Run: python src/graph_text.py")


if __name__ == "__main__":
    main()
