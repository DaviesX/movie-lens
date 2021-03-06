import numpy as np
from data import dataset
from latent import latent_truncated_svd as ltsvd
from latent import latent_dnn as ldnn
from latent import gmm
from completion import dnn_on_latent_space as dols
from eval import eval_rmse

from hparams import MOVIE_EMBEDDINGS_SIZE
from hparams import USER_EMBEDDINGS_SIZE


def main(training_mode: bool):
    np.random.seed(1331)

    # Generate raw rating datasets.
    um, row2uid, col2mid = dataset.load_user_movie_rating(
        file_name="../ml-20m/ratings.csv")
    um, row2uid, col2mid = dataset.truncate_unrated_movies(
        um=um, row2uid=row2uid, col2mid=col2mid)
    um_train, um_valid = dataset.train_and_validation_split(
        um=um, p_train=0.7)
    dataset.save_sparse_matrix(
        file="../data/raw_ratings_train.npz", mat=um_train)
    dataset.save_sparse_matrix(
        file="../data/raw_ratings_valid.npz", mat=um_valid)

    # Compute linear latent space and the corresponding vectors for both
    # users and movies.
    # Construct a latent space for movies.
    movie_embed = ltsvd.movie_latent_trunc_svd(
        um=um_train,
        embedding_size=MOVIE_EMBEDDINGS_SIZE,
        model_meta_path="../meta/latent_tsvd_movie_params.pkl",
        is_train=training_mode)
    dataset.save_dense_array(file="../data/latent_tsvd_movie_train.npz",
                             arr=movie_embed)

    movie_embed_valid = ltsvd.movie_latent_trunc_svd(
        um=um_valid,
        embedding_size=MOVIE_EMBEDDINGS_SIZE,
        model_meta_path="../meta/latent_tsvd_movie_params.pkl",
        is_train=False)
    dataset.save_dense_array(file="../data/latent_tsvd_movie_valid.npz",
                             arr=movie_embed_valid)

    # Construct a latent space for users.
    user_embed = ltsvd.user_latent_trunc_svd(
        um=um_train,
        embedding_size=USER_EMBEDDINGS_SIZE,
        model_meta_path="../meta/latent_tsvd_user_params.pkl",
        is_train=training_mode)
    dataset.save_dense_array(file="../data/latent_tsvd_user_train.npz",
                             arr=user_embed)

    user_embed_valid = ltsvd.user_latent_trunc_svd(
        um=um_valid,
        embedding_size=USER_EMBEDDINGS_SIZE,
        model_meta_path="../meta/latent_tsvd_user_params.pkl",
        is_train=False)
    dataset.save_dense_array(file="../data/latent_tsvd_user_valid.npz",
                             arr=user_embed_valid)

    num_user_clusters = gmm.search_optimal_cluster_size(
        data_set_name="user_embeddings",
        data_points=user_embed, start=1, stop=30)
    num_movie_clusters = gmm.search_optimal_cluster_size(
        data_set_name="movie_embeddings",
        data_points=movie_embed, start=1, stop=30)

    # Fine tune the latent space by using the latent_dnn model.
    num_users = row2uid.shape[0]
    num_movies = col2mid.shape[0]
    model = ldnn.latent_dnn(model_meta_path="../meta/latent_dnn.ckpt",
                            num_users=num_users,
                            num_movies=num_movies,
                            num_user_clusters=num_user_clusters,
                            num_movie_clusters=num_movie_clusters,
                            user_embed_size=USER_EMBEDDINGS_SIZE,
                            movie_embed_size=MOVIE_EMBEDDINGS_SIZE,
                            init_user_embed_table=user_embed,
                            init_movie_embed_table=movie_embed,
                            indirect_cause=True,
                            batch_size=5000,
                            num_iters=100000,
                            reset_and_train=True)
    if training_mode:
        model.fit(user_ids=um_train.row,
                  movie_ids=um_train.col,
                  ratings=um_train.data)

    user_embed, movie_embed = model.export_embeddings()
    dataset.save_dense_array(
        file="../data/latent_dnn_user.npz", arr=user_embed)
    dataset.save_dense_array(
        file="../data/latent_dnn_movie.npz", arr=movie_embed)

    pred_ratings_train = model.predict_ratings(user_ids=um_train.row,
                                               movie_ids=um_train.col)
    pred_ratings_valid = model.predict_ratings(user_ids=um_valid.row,
                                               movie_ids=um_valid.col)

    dataset.save_dense_array(
        file="../data/ldnn_preds_train.npz", arr=pred_ratings_train)
    dataset.save_dense_array(
        file="../data/ldnn_preds_valid.npz", arr=pred_ratings_valid)

    # Evaluate by the MSE objective over the embeddings.
    print("train_rmse=", eval_rmse.rmse(um_train.data, pred_ratings_train))
    print("valid_rmse=", eval_rmse.rmse(um_valid.data, pred_ratings_valid))

    # Train a missing-value completion model over the latent vectors.
    model = dols.dnn_on_latent_space(model_meta_path="../meta/dols.ckpt",
                                     user_embed_table=user_embed,
                                     movie_embed_table=movie_embed,
                                     reset_and_train=True,
                                     batch_size=5000,
                                     num_iters=100000)

    if training_mode:
        model.fit(user_ids=um_train.row,
                  movie_ids=um_train.col,
                  ratings=um_train.data)

    pred_ratings_train = model.predict(user_ids=um_train.row,
                                       movie_ids=um_train.col)
    pred_ratings_valid = model.predict(user_ids=um_valid.row,
                                       movie_ids=um_valid.col)

    dataset.save_dense_array(
        file="../data/dols_preds_train.npz", arr=pred_ratings_train)
    dataset.save_dense_array(
        file="../data/dols_preds_valid.npz", arr=pred_ratings_valid)

    # Evaluate the performance of the missing-value completion model on the
    # MSE objective.
    print("train_rmse=", eval_rmse.rmse(um_train.data, pred_ratings_train))
    print("valid_rmse=", eval_rmse.rmse(um_valid.data, pred_ratings_valid))


if __name__ == "__main__":
    main(training_mode=True)
