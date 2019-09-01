from typing import Tuple
import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd

def load_user_movie_rating(file_name: str) -> Tuple[coo_matrix, np.ndarray, np.ndarray]:
    """Load a CSV rating dataset into a numpy user-movie rating table.
    Users and movies are ordered by USER_ID and MOVIE_ID, respectively.
    However, the indices are zero-offset.

    Arguments:
        file_name {str} -- file path to the CSV dataset file.

    Returns:
        Tuple[coo_matrix, np.ndarray, np.ndarray] -- coo_matrix: a sparse U*M
                rating matrix, where U is the total number of users and M is the
                total number of movies.
            np.ndarray, np.ndarray: mappings from row and col indices to USER_ID
                and MOVIE_ID.
    """
    rating_dataset = pd.read_csv(filepath_or_buffer=file_name, sep=",")
    user_row = rating_dataset["userId"].values - 1
    movie_col = rating_dataset["movieId"].values - 1
    ratings = rating_dataset["rating"].values

    num_users = np.max(user_row) + 1
    num_movies = np.max(movie_col) + 1

    # Create rating table.
    um = coo_matrix((ratings, (user_row, movie_col)),
                    shape=(num_users, num_movies))

    # Create mappings from table row and column indices to user ID and movie ID, respectively.
    row2uid = np.arange(start=1, stop=num_users + 1, dtype=np.int32)
    col2mid = np.arange(start=1, stop=num_movies + 1, dtype=np.int32)

    return um, row2uid, col2mid

def save_sparse_matrix(file: str, mat: coo_matrix) -> None:
    np.savez_compressed(file=file,
                        data=mat.data, row=mat.row, col=mat.col,
                        shape=mat.shape)

def load_sparse_matrix(file: str) -> coo_matrix:
    with np.load(file) as loader:
        mat = coo_matrix((loader["data"], (loader["row"], loader["col"])), 
                         shape=loader["shape"])
        return mat

def truncate_unrated_movies(um: coo_matrix,
                            row2uid: np.ndarray,
                            col2mid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truncate movies that have not at all rated.

    Arguments:
        um {coo_matrix} -- U*M rating table where U is the number of users
            whereas M is the number of movies.
        row2uid {np.ndarray} -- mapping from row index to USER_ID.
        col2mid {np.ndarray} -- mapping from col index to MOVIE_ID.

    Returns:
        Tuple[coo_matrix, np.ndarray, np.ndarray] -- coo_matrix: U*M_trunc sparse
                rating tables.
            np.ndarray, np.ndarray: MOVIE_ID mapping after truncation and USER_ID
                mapping.
    """
    movie_ratings = um.tocsc()
    num_ratings_per_movie = np.sum(movie_ratings != 0, axis=0)
    movie_with_no_ratings = np.squeeze(np.asarray(num_ratings_per_movie != 0))
    return movie_ratings[:, movie_with_no_ratings].tocoo(), \
           row2uid, \
           col2mid[movie_with_no_ratings]

if __name__ == "__main__":
    """Generate raw rating datasets.
    """
    um, row2uid, col2mid = load_user_movie_rating( \
        file_name="../movie-lens-small-latest-dataset/ratings.csv")
    um, row2uid, col2mid = truncate_unrated_movies(um=um, row2uid=row2uid, col2mid=col2mid)
    # save_sparse_matrix(file="../data/raw_ratings_small.npz", mat=um)
