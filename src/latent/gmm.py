import numpy as np
from sklearn.mixture import BayesianGaussianMixture


def search_optimal_cluster_size(data_set_name: str,
                                data_points: np.ndarray,
                                start: int,
                                stop: int,
                                max_data_points=5000) -> int:
    """Determine the optimal number of clusters in the given data_points based
    on the maximum GMM log likelihood. It assumes the component in the data
    points are linearly independent. Sampling is performed to improve efficiency.

    Arguments:
        data_set_name {str} -- The name of the data set (for display purpose).
        data_points {np.ndarray} -- Of shape [num_samples, num_components].
        start {int} -- Starting number of clusters to search from.
        stop {int} -- The largest number of clusters to search on.

    Keyword Arguments:
        max_data_points {int} -- The largest number of data points to take on.
            Sampling is performed after this number of samples (default: {10000})

    Returns:
        int -- The optimal number of clusters.
    """
    if data_points.shape[0] > max_data_points:
        inds = np.arange(start=0, stop=data_points.shape[0])
        np.random.shuffle(inds)
        data_points = data_points[inds[:max_data_points], :]

    best_likelihood = -float("inf")
    best_cluster_size = start
    for i in range(start, stop + 1):
        gmm = BayesianGaussianMixture(n_components=i,
                                      covariance_type="diag",
                                      tol=1e-2,
                                      n_init=3)
        gmm.fit(X=data_points)
        if not gmm.converged_:
            continue
        likelihood = gmm.score(X=data_points)
        if likelihood > best_likelihood:
            best_likelihood = likelihood
            best_cluster_size = i
        print(data_set_name,
              "|cluster_size=", i,
              "|current_best=", best_cluster_size,
              "|current_best_score=", best_likelihood)
    return best_cluster_size
