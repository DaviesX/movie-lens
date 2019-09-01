import numpy as np
from data import dataset
from latent import latent_truncated_svd as ltsvd
from recommender import dnn_on_latent_space as dols
from eval import eval_mse

from hparams import MOVIE_EMBEDDINGS_SIZE
from hparams import USER_EMBEDDINGS_SIZE

def main(training_mode: bool):
    np.random.seed(1331)

    # Generate raw rating datasets.
    um, row2uid, col2mid = dataset.load_user_movie_rating( \
        file_name="../movie-lens-small-latest-dataset/ratings.csv")
    um, row2uid, col2mid = dataset.truncate_unrated_movies( \
        um=um, row2uid=row2uid, col2mid=col2mid)
    um_train, um_valid= dataset.train_and_validation_split( \
            um=um, p_train=0.7)
    dataset.save_sparse_matrix(file="../data/raw_ratings_train.npz", mat=um_train)
    dataset.save_sparse_matrix(file="../data/raw_ratings_valid.npz", mat=um_valid)

    # Compute linear latent space and the corresponding vectors for both
    # users and movies.
    um_train = dataset.load_sparse_matrix(file="../data/raw_ratings_train.npz")
    um_valid = dataset.load_sparse_matrix(file="../data/raw_ratings_valid.npz")

    # Construct latent space for movies.
    movie_embed_train = ltsvd.movie_latent_trunc_svd( \
        um=um_train,
        embedding_size=MOVIE_EMBEDDINGS_SIZE,
        model_meta_path="../meta/latent_tsvd_movie_params.pkl",
        is_train=True)
    dataset.save_dense_array(file="../data/latent_tsvd_movie_train.npz", 
                             arr=movie_embed_train)

    movie_embed_valid = ltsvd.movie_latent_trunc_svd( \
        um=um_valid,
        embedding_size=MOVIE_EMBEDDINGS_SIZE,
        model_meta_path="../meta/latent_tsvd_movie_params.pkl",
        is_train=False)
    dataset.save_dense_array(file="../data/latent_tsvd_movie_valid.npz",
                             arr=movie_embed_valid)

    # Construct latent space for users.
    user_embed_train = ltsvd.user_latent_trunc_svd( \
        um=um_train,
        embedding_size=USER_EMBEDDINGS_SIZE,
        model_meta_path="../meta/latent_tsvd_user_params.pkl",
        is_train=True)
    dataset.save_dense_array(file="../data/latent_tsvd_user_train.npz",
                             arr=user_embed_train)

    user_embed_valid = ltsvd.user_latent_trunc_svd( \
        um=um_valid,
        embedding_size=USER_EMBEDDINGS_SIZE,
        model_meta_path="../meta/latent_tsvd_user_params.pkl",
        is_train=False)
    dataset.save_dense_array(file="../data/latent_tsvd_user_valid.npz",
                             arr=user_embed_valid)

    # Train model using the rated embeddings training set.
    model = dols.dnn_on_latent_space( \
        model_meta_path="../meta/dols.ckpt",
        user_embed_size=USER_EMBEDDINGS_SIZE,
        movie_embed_size=MOVIE_EMBEDDINGS_SIZE,
        embedding_transform=False,
        reset_and_train=True)

    if training_mode:
        model.fit(user_embed=user_embed_train[um_train.row],
                  movie_embed=movie_embed_train[um_train.col],
                  rating=um_train.data)

    pred_ratings_train = model.predict(user_embed=user_embed_train[um_train.row],
                                       movie_embed=movie_embed_train[um_train.col])
    pred_ratings_valid = model.predict(user_embed=user_embed_train[um_valid.row],
                                       movie_embed=movie_embed_train[um_valid.col])

    dataset.save_dense_array(file="../data/dols_preds_train.npz", arr=pred_ratings_train)
    dataset.save_dense_array(file="../data/dols_preds_valid.npz", arr=pred_ratings_valid)

    # Evaluate by the MSE objective.
    print("train_mse=", eval_mse.mse(um_train.data, pred_ratings_train))
    print("valid_mse=", eval_mse.mse(um_valid.data, pred_ratings_valid))

if __name__ == "__main__":
    main(training_mode=True)
