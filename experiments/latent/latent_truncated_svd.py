import numpy as np
import pickle as pk
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD

from hparams import MOVIE_EMBEDDINGS_SIZE
from hparams import USER_EMBEDDINGS_SIZE
from dataset import load_sparse_matrix

def movie_latent_trunc_svd(um: coo_matrix,
                           embedding_size: int,
                           model_meta_path: str,
                           is_train: bool) -> np.ndarray:
    """Compute a low dimensional representation vector for each movie.
    The representation is obtained by performing a sparse linear PCA (i.e. truncated SVD).

    Arguments:
        um {coo_matrix} -- A U*M rating table where U is the number of users,
            and M is the the number of movies.
        embedding_size {int} -- Target dimension to reduce towards.
            This value must be less than U and greater than 0.
        model_meta_path {str} -- Location to store the parameters of this embedding model.

    Returns:
        np.ndarray -- Returns an M*E matrix where E is the embedding size.
    """
    model = None
    if is_train:
        model = TruncatedSVD(embedding_size)
        model.fit(X=um.T)
        pk.dump(model, open(model_meta_path, "wb"))
    else:
        model = pk.load(file=open(model_meta_path, "rb"))
    movie_embed = model.transform(X=um.T)
    return movie_embed

def user_latent_trunc_svd(um: coo_matrix,
                          embedding_size: int,
                          model_meta_path: str,
                          is_train: bool) -> np.ndarray:
    """Compute a low dimensional representation vector for each user.
    The representation is obtained by performing a sparse linear PCA (i.e. truncated SVD).

    Arguments:
        um {coo_matrix} -- A U*M rating table where U is the number of users,
            and M is the the number of movies.
        embedding_size {int} -- Target dimension to reduce towards.
            This value must be less than M and greater than 0.
        model_meta_path {str} -- Location to store the parameters of this embedding model.

    Returns:
        np.ndarray -- Returns an U*E matrix where E is the embedding size.
    """
    model = None
    if is_train:
        model = TruncatedSVD(embedding_size)
        model.fit(X=um)
        pk.dump(model, open(model_meta_path, "wb"))
    else:
        model = pk.load(file=open(model_meta_path, "rb"))
    user_embed = model.transform(X=um)
    return user_embed

if __name__ == "__main__":
    """Compute linear latent space and the corresponding vectors for
    both users and movies.
    """
    um_train = load_sparse_matrix(file="../data/raw_ratings_train.npz")
    um_valid = load_sparse_matrix(file="../data/raw_ratings_valid.npz")

    # Construct latent space for movies.
    movie_embed_train = movie_latent_trunc_svd( \
        um=um_train,
        embedding_size=MOVIE_EMBEDDINGS_SIZE,
        model_meta_path="../meta/latent_tsvd_movie_params.pkl",
        is_train=True)
    np.savez_compressed(file="../data/latent_tsvd_movie_train.npz", arr=movie_embed_train)

    movie_embed_valid = movie_latent_trunc_svd( \
        um=um_valid,
        embedding_size=MOVIE_EMBEDDINGS_SIZE,
        model_meta_path="../meta/latent_tsvd_movie_params.pkl",
        is_train=False)
    np.savez_compressed(file="../data/latent_tsvd_movie_valid.npz", arr=movie_embed_valid)

    # Construct latent space for users.
    user_embed_train = user_latent_trunc_svd( \
        um=um_train,
        embedding_size=USER_EMBEDDINGS_SIZE,
        model_meta_path="../meta/latent_tsvd_user_params.pkl",
        is_train=True)
    np.savez_compressed(file="../data/latent_tsvd_user_train.npz", arr=user_embed_train)

    user_embed_valid = user_latent_trunc_svd( \
        um=um_valid,
        embedding_size=USER_EMBEDDINGS_SIZE,
        model_meta_path="../meta/latent_tsvd_user_params.pkl",
        is_train=False)
    np.savez_compressed(file="../data/latent_tsvd_user_valid.npz", arr=user_embed_valid)
