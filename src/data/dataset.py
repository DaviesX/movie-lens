from typing import Tuple
import random as rnd
import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd

def load_sparse_matrix(file: str) -> coo_matrix:
    with np.load(file) as loader:
        mat = coo_matrix((loader["data"], (loader["row"], loader["col"])), 
                         shape=loader["shape"])
        return mat

def save_sparse_matrix(file: str, mat: coo_matrix) -> None:
    np.savez_compressed(file=file,
                        data=mat.data, row=mat.row, col=mat.col,
                        shape=mat.shape)

def load_dense_array(file: str) -> np.ndarray:
    with np.load(file) as loader:
        return loader["arr"]

def save_dense_array(file: str, arr: np.ndarray) -> None:
    np.savez_compressed(file=file, arr=arr)

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
                    shape=(num_users, num_movies),
                    dtype=np.float32)
    um.row = um.row.astype(np.int32)
    um.col = um.col.astype(np.int32)

    # Create mappings from table row and column indices to user ID and movie ID, respectively.
    row2uid = np.arange(start=1, stop=num_users + 1, dtype=np.int32)
    col2mid = np.arange(start=1, stop=num_movies + 1, dtype=np.int32)

    return um, row2uid, col2mid

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

def train_and_validation_split(um: coo_matrix,
                               p_train: float) -> \
    Tuple[coo_matrix, coo_matrix, np.ndarray, np.ndarray]:
    """Randomly split the U*M ratings table into training and hold-out sets.
    hold-out entry selection is done by randomly mask out movie-user pairs.

    Arguments:
        um {coo_matrix} -- U*M rating table where U is the number of users
            whereas M is the number of movies.
        p_train {float} -- Proportion in which the users will be partitioned
            as the training set.

    Returns:
        Tuple[coo_matrix, coo_matrix, np.ndarray, np.ndarray] --
            coo_matrix, coo_matrix: A tuple of (U*M, U*M) training and
                validation rating tables, respectively.
    """
    # First shuffle entries (indices) in the user-movie matrix table
    inds = np.arange(start=0, stop=um.data.shape[0])
    rnd.shuffle(x=inds)

    # Take the first p_train*dataset_size of data out as the training set,
    # and keep the rest for validation.
    train_upper_bound = int(round(p_train*um.data.shape[0]))
    um_train = coo_matrix((um.data[inds[:train_upper_bound]],
                           (um.row[inds[:train_upper_bound]],
                            um.col[inds[:train_upper_bound]])),
                           shape=um.shape)
    um_train.row = um_train.row.astype(np.int32)
    um_train.col = um_train.col.astype(np.int32)

    um_valid = coo_matrix((um.data[inds[train_upper_bound:]],
                           (um.row[inds[train_upper_bound:]],
                            um.col[inds[train_upper_bound:]])),
                           shape=um.shape)
    um_valid.row = um_valid.row.astype(np.int32)
    um_valid.col = um_valid.col.astype(np.int32)

    return um_train, um_valid
