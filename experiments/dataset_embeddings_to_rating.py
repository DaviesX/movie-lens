import numpy as np

def embeddings_to_rating(user_embed: np.ndarray, 
                         movie_embed: np.ndarray, 
                         user_ids: np.ndarray,
                         movie_ids: np.ndarray,
                         um: np.ndarray) -> np.ndarray:
    """Generate a dataset for all rated user-movie pairs using embeddings.
    The dataset will have the format: [user_id, movie_id, user_embed, movie_embed, rating].
    
    Arguments:
        user_embed {np.ndarray} -- A U*E_user table where U is the number of users, 
            and E_user is the user embedding size.
        movie_embed {np.ndarray} -- A M*E_movie table where M is the number of movies,
            and E_movie is the movie embedding size.
        user_ids {np.ndarray} -- U number of user identifiers 
            each maps to the user_embed entries.
        movie_ids {np.ndarray} -- M number of movie identifiers 
            each maps to the movie_embed entries.
        um {np.ndarray} -- U*M rating table where U is the number of users, 
            and M is the number of movies.
    
    Returns:
        np.ndarray -- it has the format: format: 
            [user_id, movie_id, user_embed, movie_embed, rating]
            Therefore the dimension at each row is: 
            user_embed.shape[1] + movie_embed.shape[1] + 3
            The number of rows will be total number of ratings.
    """
    rated_pairs = np.where(um != -1)
    num_records = rated_pairs[0].shape[0]
    row_size = user_embed.shape[1] + movie_embed.shape[1] + 3

    dataset = np.zeros((num_records, row_size))
    i_record = 0
    for user_ind, movie_ind in zip(*rated_pairs):
        dataset[i_record, 0] = user_ids[user_ind]
        dataset[i_record, 1] = movie_ids[movie_ind]
        dataset[i_record, 2:2 + user_embed.shape[1]] = user_embed[user_ind, :]
        dataset[i_record, 2 + user_embed.shape[1]:\
                2 + user_embed.shape[1] + movie_embed.shape[1]] = movie_embed[movie_ind, :]
        dataset[i_record, -1] = um[user_ind, movie_ind]
        i_record += 1
    return dataset

if __name__ == "__main__":
    um_train = np.load(file="../data/ratings_small_train.npy")
    user_embed_train = np.load(file="../data/embeddings_user_train.npy")
    movie_embed_train = np.load(file="../data/embeddings_movie_train.npy")
    user_ids = np.load(file="../data/trunc_ratings_small_user_ids.npy")
    movie_ids = np.load(file="../data/trunc_ratings_small_movie_ids.npy")

    dataset_train = embeddings_to_rating(user_embed_train, 
                                         movie_embed_train, 
                                         user_ids, 
                                         movie_ids, 
                                         um_train)

    np.save(file="../data/rated_embeddings_train.npy", arr=dataset_train)

    um_valid = np.load(file="../data/ratings_small_validation.npy")
    user_embed_valid = np.load(file="../data/embeddings_user_valid.npy")
    movie_embed_valid = np.load(file="../data/embeddings_movie_valid.npy")

    dataset_valid = embeddings_to_rating(user_embed_valid, 
                                         movie_embed_valid, 
                                         user_ids, 
                                         movie_ids, 
                                         um_valid)

    np.save(file="../data/rated_embeddings_valid.npy", arr=dataset_valid)