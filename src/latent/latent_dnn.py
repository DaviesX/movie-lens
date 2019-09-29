from typing import Tuple, List
import numpy as np
import tensorflow as tf

from utils import nnutils
from latent import gmm


def sample_batch(user_ids: np.ndarray,
                 movie_ids: np.ndarray,
                 ratings: np.ndarray,
                 batch_size: int) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[summary]

    Returns:
        [type] -- [description]
    """
    dataset_size = ratings.shape[0]
    batch_idx = np.random.randint(low=0, high=dataset_size,
                                  size=batch_size)
    batch_user_ids = user_ids[batch_idx]
    batch_movie_ids = movie_ids[batch_idx]
    batch_ratings = ratings[batch_idx]

    return batch_user_ids, batch_movie_ids, batch_ratings


class latent_dnn:
    def __init__(self,
                 model_meta_path: str,
                 num_users: int,
                 num_movies: int,
                 num_user_clusters: int,
                 num_movie_clusters: int,
                 user_embed_size: int,
                 movie_embed_size: int,
                 init_user_embed_table: np.ndarray,
                 init_movie_embed_table: np.ndarray,
                 indirect_cause: bool,
                 reset_and_train: bool,
                 num_iters=10000,
                 batch_size=400,
                 learning_rate=0.001):
        """Construct latent spaces for both user and movie by using rating
        prediction as the training task.

        Arguments:
            model_meta_path {str} -- Location where the model parameters
                are stored/going to be stored.
            num_users {int} -- The total number of unique users.
            num_movies {int} -- The total number of unique movies.
            num_user_clusters {int} -- Suggested number of gaussian user
                clusters.
            num_movie_clusters {int} -- Suggested number of gaussian movie
                clusters.
            user_embed_size {int} -- The size of user embeddings vector to
                be generated.
            movie_embed_size {int} -- The size of movie embeddings vector
                to be generated.
            init_user_embed_table {np.ndarray} -- Optional, initial embeddings
                vector for each user.
            init_movie_embed_table {np.ndarray} -- Optional, initial embeddings
                vector for each movie.
            indirect_cause {bool} -- Whether to assume there is an indirect
                cause from the latent variables to the observable outcome.
            reset_and_train {bool} -- Whether to reset model parameters or
                load them from checkpoint files.

        Keyword Arguments:
            num_iters {int} -- The maximum number of iterations over batches
                is going to perform (default: {20000})
            batch_size {int} -- How many training examples are going to
                feed for each weights update. (default: {100})
            learning_rate {float} -- The velocity of each gradient descent
                step. (default: {0.001})
        """
        # Configurable hyper-params
        self.user_embed_size_ = user_embed_size
        self.movie_embed_size_ = movie_embed_size
        self.model_meta_path_ = model_meta_path
        self.reset_and_train_ = reset_and_train
        self.num_iters_ = num_iters
        self.batch_size_ = batch_size
        self.indirect_cause_ = indirect_cause
        self.learning_rate_ = learning_rate

        self.num_users_ = num_users
        self.num_movies_ = num_movies

        # Model varibles
        self.user_embed_table_ = tf.Variable(
            initial_value=tf.random.truncated_normal(shape=(num_users, user_embed_size)))
        self.movie_embed_table_ = tf.Variable(
            initial_value=tf.random.truncated_normal(shape=(num_movies, movie_embed_size)))

        if init_user_embed_table is not None:
            self.user_embed_table_.assign(init_user_embed_table)

        if init_movie_embed_table is not None:
            self.movie_embed_table_.assign(init_movie_embed_table)

        self.ratings_t1i_ = nnutils.transform(input_size=user_embed_size + movie_embed_size,
                                              output_size=(user_embed_size + movie_embed_size)//2)
        self.ratings_t2i_ = nnutils.transform(input_size=(
            user_embed_size + movie_embed_size)//2, output_size=1)
        self.ratings_t1_ = nnutils.transform(
            input_size=user_embed_size + movie_embed_size, output_size=1)

        self.user_gmm_ = gmm.gmm_likelihood(num_clusters=num_user_clusters)
        self.movie_gmm_ = gmm.gmm_likelihood(num_clusters=num_movie_clusters)

        # Variable sets
        self.regi_vars_ = [self.ratings_t1i_.weights()]
        self.reg_vars_ = [self.ratings_t1_.weights()]

        self.ratings_modeli_vars_ = list(nnutils.collect_transform_vars(
            [self.ratings_t1i_, self.ratings_t2i_]).values())
        self.ratings_model_vars_ = list(nnutils.collect_transform_vars(
            [self.ratings_t1_]).values())

        # Create model checkpoint
        model_vars = nnutils.collect_transform_vars(
            [self.ratings_t1i_, self.ratings_t2i_, self.ratings_t1_])
        model_vars["user_embed_table"] = self.user_embed_table_
        model_vars["movie_embed_table"] = self.movie_embed_table_
        self.ckpt_ = tf.train.Checkpoint(**model_vars)

        if not reset_and_train:
            status = self.ckpt_.restore(
                tf.train.latest_checkpoint(self.model_meta_path_))
            status.assert_consumed()

    @tf.function
    def predict_ratings(self,
                        user_ids: np.ndarray,
                        movie_ids: np.ndarray,
                        drop_prob=0.0) -> np.ndarray:
        """Predict the rating value of each (user_id, movie_id) pair.

        Arguments:
            user_ids {np.ndarray} -- [description]
            movie_ids {np.ndarray} -- [description]

        Keyword Arguments:
            drop_prob {float} -- The rating of dropping out perceptron units.
                (default: {0.0})

        Returns:
            np.ndarray -- [description]
        """
        user_embed = tf.nn.embedding_lookup(
            params=self.user_embed_table_, ids=user_ids)
        movie_embed = tf.nn.embedding_lookup(
            params=self.movie_embed_table_, ids=movie_ids)

        concat_features = tf.concat(values=[user_embed, movie_embed], axis=1)
        concat_features = tf.nn.dropout(x=concat_features, rate=drop_prob)
        concat_features = tf.transpose(a=concat_features)

        preds = None
        if self.indirect_cause_:
            features = self.ratings_t1i_(x=concat_features, act_fn="tanh")
            preds = 2.5 * \
                (self.ratings_t2i_(x=features, act_fn="tanh") + 1)
        else:
            preds = 2.5 * \
                (self.ratings_t1_(x=concat_features, act_fn="tanh") + 1)

        return preds[0, :], user_embed, movie_embed

    def fit(self,
            user_ids: np.ndarray,
            movie_ids: np.ndarray,
            ratings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the NN model to construct latent spaces for both user and movie
        by using rating prediction as the training task.

        Arguments:
            user_ids {np.ndarray} -- A list of user ids pairing up with the
                movie_ids.
            movie_ids {np.ndarray} -- A list of movie ids pairing up with the
                user_ids.
            rating {np.ndarray} -- a float array consists of ratings for each
                user-movie pair record.
        Note:
            fit() assumes that user_embed.shape[0] == movie_embed.shape[0] and
            movie_embed.shape[0] == rating.shape[0]
        """
        optimizer = tf.optimizers.Adamax(learning_rate=self.learning_rate_)

        curr_task = "rating_pred"
        num_iters_for_curr_task = 0
        for i in range(self.num_iters_):
            if num_iters_for_curr_task > 1000 and curr_task == "embed_tune":
                curr_task = "rating_pred"
                num_iters_for_curr_task = 0
            if num_iters_for_curr_task > 1000 and curr_task == "rating_pred":
                curr_task = "embed_tune"
                num_iters_for_curr_task = 0

            batch_user_ids, batch_movie_ids, batch_ratings = \
                sample_batch(user_ids=user_ids,
                             movie_ids=movie_ids,
                             ratings=ratings,
                             batch_size=self.batch_size_)

            if curr_task == "embed_tune":
                if num_iters_for_curr_task == 0:
                    print("Finding MLE for user GMM...")
                    self.user_gmm_.mle(x=self.user_embed_table_.numpy())

                    print("Finding MLE for movie GMM...")
                    self.movie_gmm_.mle(x=self.movie_embed_table_.numpy())
                with tf.GradientTape() as tape:
                    ratings_hat, user_embed, movie_embed = self.predict_ratings(
                        user_ids=batch_user_ids, movie_ids=batch_movie_ids, drop_prob=0.1)
                    l_ratings = tf.metrics.mean_squared_error(
                        y_true=batch_ratings, y_pred=ratings_hat)
                    l_user_gmm = -self.user_gmm_.likelihood_score(x=user_embed)
                    l_movie_gmm = - \
                        self.movie_gmm_.likelihood_score(x=movie_embed)
                    loss = l_ratings + 0.1*l_user_gmm + 0.1*l_movie_gmm

                    if i % 100 == 0:
                        print("Epoch", round(i*self.batch_size_/ratings.shape[0], 3),
                              "|task=embed_tune",
                              "|l_ratings=", float(l_ratings),
                              "|l_user_gmm=", float(l_user_gmm),
                              "|l_movie_gmm=", float(l_movie_gmm),
                              "|loss=", float(loss))

                    task_vars = [self.user_embed_table_,
                                 self.movie_embed_table_]
                    grads = tape.gradient(target=loss, sources=task_vars)
                    optimizer.apply_gradients(
                        grads_and_vars=zip(grads, task_vars))
            elif curr_task == "rating_pred":
                with tf.GradientTape() as tape:
                    ratings_hat, _, _ = self.predict_ratings(
                        user_ids=batch_user_ids, movie_ids=batch_movie_ids, drop_prob=0.3)
                    l_ratings = tf.metrics.mean_squared_error(
                        y_true=batch_ratings, y_pred=ratings_hat)
                    l_reg = nnutils.regularizer_loss(
                        weights=self.regi_vars_ if self.indirect_cause_
                        else self.reg_vars_, alpha=0.01)

                    loss = l_ratings + l_reg

                    if i % 100 == 0:
                        print("Epoch", round(i*self.batch_size_/ratings.shape[0], 3),
                              "|task=rating_pred",
                              "|l_ratings=", float(l_ratings),
                              "|l_reg=", float(l_reg),
                              "|loss=", float(loss))

                    task_vars = self.ratings_modeli_vars_ if self.indirect_cause_ \
                        else self.ratings_model_vars_
                    grads = tape.gradient(target=loss, sources=task_vars)
                    optimizer.apply_gradients(
                        grads_and_vars=zip(grads, task_vars))

            num_iters_for_curr_task += 1

        print("Training complete. ")
        self.ckpt_.save(self.model_meta_path_ +
                        "/latent_dnn_i=" + str(self.indirect_cause_))
        print("Saved model parameters.")

    def export_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Export the fitted user and movie embeddings.

        Returns:
            Tuple[np.ndarray, np.ndarray] -- A tuple of user embeddings and
                movie embeddings
        """
        return self.user_embed_table_.numpy(), self.movie_embed_table_.numpy()
