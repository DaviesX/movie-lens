from typing import Tuple
import numpy as np
import random as rnd

def train_and_validation_split_by_users(um: np.ndarray, p_train: float) -> \
    Tuple[np.ndarray, np.ndarray]:
    """Randomly split the U*M ratings table into training and hold-out sets.
    hold-out entry selection is done by randomly mask out movie-user pairs.
    
    Arguments:
        um {np.ndarray} -- U*M rating table where U is the number of users 
            whereas M is the number of movies.
        p_train {float} -- Proportion in which the users will be 
            partitioned as the training set.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] -- A tuple of
            (U_train*M, U_val*M) rating tables
    """
    rated_pairs = np.where(um != -1)

    num_ratings = rated_pairs[0].shape[0]
    pair_inds = np.arange(start=0, stop=num_ratings)
    rnd.shuffle(x=pair_inds)

    training_inds = pair_inds[:int(num_ratings*p_train)]
    valid_inds = pair_inds[int(num_ratings*p_train):]
    training_pairs = rated_pairs[0][training_inds], rated_pairs[1][training_inds]
    valid_pairs = rated_pairs[0][valid_inds], rated_pairs[1][valid_inds]
    
    um_train = np.zeros(um.shape)
    um_train.fill(-1.0)
    for movie_ind, user_ind in zip(*training_pairs):
        um_train[movie_ind, user_ind] = um[movie_ind, user_ind]

    um_valid = np.zeros(um.shape)
    um_valid.fill(-1.0)
    for movie_ind, user_ind in zip(*valid_pairs):
        um_valid[movie_ind, user_ind] = um[movie_ind, user_ind]

    return um_train, um_valid

if __name__ == "__main__":
    """Generate training and validation split from the truncated dataset.
    """
    um = np.load(file="../data/trunc_ratings_small.npy")

    um_train, um_valid = \
        train_and_validation_split_by_users(um=um, p_train=0.8)
    
    np.save(file="../data/ratings_small_train.npy", arr=um_train)
    np.save(file="../data/ratings_small_validation.npy", arr=um_valid)