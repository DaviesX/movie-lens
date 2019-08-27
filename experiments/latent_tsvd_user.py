import numpy as np
import pickle as pk
from sklearn.decomposition import TruncatedSVD

from hparams import USER_EMBEDDINGS_SIZE

def user_latent_trunc_svd(um: np.ndarray, 
                          embedding_size: int, 
                          model_meta_path: str,
                          is_train: bool) -> np.ndarray:
    """Compute a low dimensional representation vector for each user. 
    The representation is obtained by performing a sparse linear PCA (i.e. truncated SVD).
    
    Arguments:
        um {np.ndarray} -- A U*M rating table where U is the number of users, 
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
    um_embedded = model.transform(X=um)
    return um_embedded

if __name__ == "__main__":
    """Generate user embeddings for each user.
    """
    um_train = np.load(file="../data/ratings_small_train.npy")
    um_embedded_train = user_latent_trunc_svd(um=um_train, embedding_size=USER_EMBEDDINGS_SIZE, \
        model_meta_path="../meta/embeddings_tsvd_users_params.dump",
        is_train=True)
    np.save(file="../data/embeddings_user_train.npy", arr=um_embedded_train)

    um_valid = np.load(file="../data/ratings_small_validation.npy")
    um_embedded_valid = user_latent_trunc_svd(um=um_valid, embedding_size=USER_EMBEDDINGS_SIZE, \
        model_meta_path="../meta/embeddings_tsvd_users_params.dump",
        is_train=False)
    np.save(file="../data/embeddings_user_valid.npy", arr=um_embedded_valid)