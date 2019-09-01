import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd

def load_user_movie_rating(file_name: str) -> coo_matrix:
    """Load a CSV rating dataset into a numpy user-movie rating table.
    Users and movies are ordered by USER_ID and MOVIE_ID, respectively.
    However, the indices are zero-offset.

    Arguments:
        file_name {str} -- file path to the CSV dataset file.

    Returns:
        coo_matrix -- a sparse U*M rating matrix, where U is the total number
            of users and M is the total number of movies.
    """
    rating_dataset = pd.read_csv(filepath_or_buffer=file_name, sep=",")
    user_row = rating_dataset["userId"].values - 1
    movie_col = rating_dataset["movieId"].values - 1
    ratings = rating_dataset["rating"].values

    num_users = np.max(user_row) + 1
    num_movies = np.max(movie_col) + 1

    um = coo_matrix((ratings, (user_row, movie_col)), shape=(num_users, num_movies))
    return um

def save_sparse_matrix(file: str, mat: coo_matrix) -> None:
    np.savez_compressed(file=file,
                        data=mat.data, row=mat.row, col=mat.col,
                        shape=mat.shape)

def load_sparse_matrix(file: str) -> coo_matrix:
    with np.load(file) as loader:
        mat = coo_matrix((loader["data"], (loader["row"], loader["col"])), 
                         shape=loader["shape"])
        return mat

if __name__ == "__main__":
    """Generate raw rating datasets.
    """
    um = load_user_movie_rating(file_name="../movie-lens-small-latest-dataset/ratings.csv")
    save_sparse_matrix(file="../data/raw_ratings_small.npz", mat=um)
