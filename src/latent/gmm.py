import numpy as np
import tensorflow as tf
from sklearn.mixture import BayesianGaussianMixture


def search_optimal_cluster_size(data_set_name: str,
                                data_points: np.ndarray,
                                start: int,
                                stop: int,
                                max_data_points=2000) -> int:
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
            Sampling is performed after this number of samples (default: {2000})

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


class gmm_likelihood:
    def __init__(self, num_clusters: int, max_data_points=10000):
        """[summary]

        Arguments:
            num_clusters {int} -- [description]

        Keyword Arguments:
            max_data_points {int} -- [description] (default: {10000})
        """
        self.num_clusters_ = num_clusters
        self.max_data_points_ = max_data_points
        self.gmm_ = BayesianGaussianMixture(n_components=self.num_clusters_,
                                            covariance_type="diag",
                                            warm_start=True,
                                            tol=1e-2,
                                            n_init=10)

    def mle(self, x: np.ndarray) -> None:
        """[summary]

        Arguments:
            x {np.ndarray} -- [description]

        Returns:
            None -- [description]
        """
        if x.shape[0] > self.max_data_points_:
            inds = np.arange(start=0, stop=x.shape[0])
            np.random.shuffle(inds)
            x = x[inds[:self.max_data_points_], :]
        self.gmm_.fit(X=x)

    @tf.function(experimental_relax_shapes=True)
    def component_log_density_(self, i: int, x: np.ndarray) -> np.ndarray:
        """Returns the log density of x at cluster i. The log density is
        computed as follow:

        log(P(x[:, 0:M])) = sum_j(log(t[i]) + log(N(x[j] | mu[i, j], sig[i, j])))

        Arguments:
            i {int} -- Evaluate on the ith cluster component.
            x {np.ndarray} -- Of shape [num_samples, num_features]

        Returns:
            np.ndarray -- Of shape [num_samples]
        """
        num_components = self.gmm_.means_.shape[1]
        mu = np.reshape(self.gmm_.means_[i, :], (1, num_components))
        sig2 = np.reshape(self.gmm_.covariances_[i, :] *
                          self.gmm_.covariances_[i, :], (1, num_components))
        dev = x - mu
        exponent = -dev*dev/(2.0*sig2)
        log_normalizer = -0.5*tf.math.log(np.pi*sig2 + 0.001)
        component_log_pdf = tf.dtypes.cast(
            log_normalizer, tf.float32) + exponent
        log_ti = tf.math.log(self.gmm_.weights_[i] + 0.001)
        log_pdf = tf.dtypes.cast(log_ti, tf.float32) + \
            tf.reduce_sum(component_log_pdf, axis=1)
        return log_pdf

    @tf.function(experimental_relax_shapes=True)
    def likelihood_score(self, x: np.ndarray) -> float:
        """Compute the per-sample averaged complete-data likelihood.

        Arguments:
            x {np.ndarray} -- Of shape [num_samples, num_features]

        Returns:
            float -- The averaged likelihood.
        """
        dens = [tf.math.exp(self.component_log_density_(
            i, x)) for i in range(self.num_clusters_)]
        gmm_log_dens = tf.math.log(sum(dens) + 0.001)
        likelihood = tf.reduce_mean(gmm_log_dens)
        return likelihood
