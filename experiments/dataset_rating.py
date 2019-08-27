import numpy as np
import pandas as pd

def load_user_movie_rating(file_name: str) -> np.ndarray:
    """Load a CSV rating dataset into a numpy user-movie rating table.
    Users and movies are ordered by USER_ID and MOVIE_ID, respectively.
    However, the indices are zero-offset.
    
    Arguments:
        file_name {str} -- file path to the CSV dataset file.
    
    Returns:
        np.ndarray -- a dense U*M rating matrix, where U is the total number 
            of users and M is the total number of movies. 
    """
    rating_dataset = pd.read_csv(filepath_or_buffer=file_name, sep=",")
    user_to_movie_inds = rating_dataset[["userId", "movieId"]].values
    ratings = rating_dataset["rating"].values

    num_users = np.max(user_to_movie_inds[:,0])
    num_movies = np.max(user_to_movie_inds[:,1])

    um = np.zeros((num_users, num_movies))
    um.fill(-1.0)
    um[(user_to_movie_inds[:,0] - 1, user_to_movie_inds[:,1] - 1)] = ratings
    return um



if __name__ == "__main__":
    """Generate raw rating datasets.
    """
    um = load_user_movie_rating(file_name="../movie-lens-small-latest-dataset/ratings.csv")
    np.save(file="../data/raw_ratings_small.npy", arr=um)
