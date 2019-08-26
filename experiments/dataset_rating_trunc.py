from typing import Tuple
import numpy as np

def truncate_unrated_movies(um: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truncate movies that have not at all rated.
    
    Arguments:
        um {np.ndarray} -- U*M rating table where U is the number of users 
            whereas M is the number of movies.
    
    Returns:
        Tuple[np.ndarray, np.ndarray] -- U*M_trunc rating tables and MOVIE_ID mapping after truncation.
    """
    num_users = um.shape[0]
    user_ids = np.arange(start=1, stop=num_users+1, step=1)

    num_movies = um.shape[1]
    movie_ids = np.arange(start=1, stop=num_movies+1, step=1)

    num_ratings_per_movie = np.sum(um != -1, axis=0)
    return um[:,num_ratings_per_movie != 0], \
           movie_ids[num_ratings_per_movie != 0], \
           user_ids


if __name__ == "__main__":
    um = np.load(file="../data/raw_ratings_small.npy")
    trunc, movie_ids, user_ids = truncate_unrated_movies(um=um)
    np.save(file="../data/trunc_ratings_small.npy", arr=trunc)
    np.save(file="../data/trunc_ratings_small_movie_ids.npy", arr=movie_ids)
    np.save(file="../data/trunc_ratings_small_user_ids.npy", arr=user_ids)