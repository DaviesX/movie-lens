import numpy as np
import pickle as pk
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD

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
