import numpy as np

def mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the mse loss.
    
    Arguments:
        y {[np.ndarray]} -- 1D float values.
        y_pred {[np.ndarray]} -- 1D float values.
    
    Returns:
        float -- MSE value.
    """
    d = (y_pred - y)
    return np.mean(d*d)

if __name__ == "__main__":
    ground_truth_train = np.load(file="../data/rated_embeddings_train.npy")
    rating_pred_train = np.load(file="../data/rating_pred_train_nn.npy")
    print("train_mse=", mse(rating_pred_train[:,0], ground_truth_train[:,-1]))

    ground_truth_valid = np.load(file="../data/rated_embeddings_valid.npy")
    rating_pred_valid = np.load(file="../data/rating_pred_valid_nn.npy")
    print("valid_mse=", mse(rating_pred_valid[:,0], ground_truth_valid[:,-1]))
