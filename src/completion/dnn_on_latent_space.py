from typing import Tuple, List
import tensorflow as tf
import numpy as np

from utils import nnutils


def sample_batch(user_embed: np.ndarray,
                 movie_embed: np.ndarray,
                 ratings: np.ndarray,
                 batch_size: int) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[summary]

    Returns:
        [Tuple[np.ndarray, np.ndarray, np.ndarray]] -- [description]
    """
    dataset_size = ratings.shape[0]
    batch_idx = np.random.randint(low=0, high=dataset_size,
                                  size=batch_size)
    batch_user_embed = user_embed[batch_idx]
    batch_movie_embed = movie_embed[batch_idx]
    batch_ratings = ratings[batch_idx]

    return batch_user_embed, batch_movie_embed, batch_ratings


class dnn_on_latent_space:
    def __init__(self,
                 model_meta_path: str,
                 user_embed_size: int,
                 movie_embed_size: int,
                 reset_and_train: bool,
                 num_iters=200000,
                 batch_size=400,
                 learning_rate=0.001):
        """Construct a deep neural network missing-value completion model based on
        a latent space that explains users and movies.

        Arguments:
            model_meta_path {str} -- Location where the model parameters
                are stored/going to be stored.
            user_embed_size {int} -- The size of user embeddings vector.
            movie_embed_size {int} -- The size of movie embeddings vector.
            embedding_transform {bool} -- Whether to transform embeddings
                before feature compression.
            reset_and_train {bool} -- Whether to reset model parameters or
                load them from checkpoint files.

        Keyword Arguments:
            num_iters {int} -- The maximum number of iterations over batches
                is going to perform (default: {20000})
            batch_size {int} -- How many training examples are going to
                feed for each weights update. (default: {100})
            learning_rate {float} -- The velocity of each gradient descent step.
                (default: {0.001})
        """
        # Configurable hyper-params
        self.user_embed_size_ = user_embed_size
        self.movie_embed_size_ = movie_embed_size
        self.model_meta_path_ = model_meta_path
        self.reset_and_train_ = reset_and_train
        self.num_iters_ = num_iters
        self.batch_size_ = batch_size
        self.learning_rate_ = learning_rate

        # Model varibles
        self.ratings_t1i_ = nnutils.transform(input_size=user_embed_size + movie_embed_size,
                                              output_size=(user_embed_size + movie_embed_size)//2)
        self.ratings_t2i_ = nnutils.transform(input_size=(
            user_embed_size + movie_embed_size)//2, output_size=1)

        # Variable sets
        self.regi_vars_ = [self.ratings_t1i_.weights()]
        self.modeli_vars_ = list(nnutils.collect_transform_vars(
            [self.ratings_t1i_, self.ratings_t2i_]).values())

        # Create model checkpoint
        self.ckpt_ = tf.train.Checkpoint(**nnutils.collect_transform_vars(
            [self.ratings_t1i_, self.ratings_t2i_]))
        if not reset_and_train:
            status = self.ckpt_.restore(
                tf.train.latest_checkpoint(self.model_meta_path_))
            status.assert_consumed()

    @tf.function
    def predict(self,
                user_embed: np.ndarray,
                movie_embed: np.ndarray,
                drop_prob=0.0) -> np.ndarray:
        """Predict the rating value of each (user_id, movie_id) pair by using
        their representation vector pair (user_embed, movie_embed).

        Arguments:
            user_embed {np.ndarray} -- U*E_user matrix where U is
                the number of users and E_user is the embedding size.
            movie_embed {np.ndarray} -- M*E_movie matrix where M is
                the number of movies and E_movie is the embedding size.

        Keyword Arguments:
            drop_prob {float} -- The rating of dropping out perceptron units.
                (default: {0.0})

        Returns:
            np.ndarray -- [description]
        """
        concat_features = tf.concat(values=[user_embed, movie_embed], axis=1)
        concat_features = tf.nn.dropout(x=concat_features, rate=drop_prob)
        concat_features = tf.transpose(a=concat_features)

        features = self.ratings_t1i_(x=concat_features, act_fn="tanh")
        preds = 2.5 * \
            (self.ratings_t2i_(x=features, act_fn="tanh") + 1)

        return preds[0, :]

    def fit(self,
            user_embed: np.ndarray,
            movie_embed: np.ndarray,
            ratings: np.ndarray) -> None:
        """Fit the NN model to the user-movie embedding pairs and ratings.

        Arguments:
            user_embed {np.ndarray} -- U*E_user matrix where U is
                the number of users and E_user is the embedding size.
            movie_embed {np.ndarray} -- M*E_movie matrix where M is
                the number of movies and E_movie is the embedding size.
            ratings {np.ndarray} -- a float array consists of ratings for each
                user-movie pair record.
        Note:
            fit() assumes that user_embed.shape[0] == movie_embed.shape[0] and
            movie_embed.shape[0] == rating.shape[0]
        """
        optimizer = tf.optimizers.Adamax(learning_rate=self.learning_rate_)

        for i in range(self.num_iters_):
            batch_user_embed, batch_movie_embed, batch_ratings = \
                sample_batch(user_embed=user_embed,
                             movie_embed=movie_embed,
                             ratings=ratings,
                             batch_size=self.batch_size_)

            with tf.GradientTape() as tape:
                ratings_hat = self.predict(
                    user_embed=batch_user_embed, movie_embed=batch_movie_embed, drop_prob=0.3)
                l_ratings = tf.metrics.mean_squared_error(
                    y_true=batch_ratings, y_pred=ratings_hat)
                l_reg = nnutils.regularizer_loss(weights=self.regi_vars_)
                loss = l_ratings + l_reg

                if i % 1000 == 0:
                    print("Epoch", round(i*self.batch_size_/ratings.shape[0], 3),
                          "|l_ratings=", float(l_ratings),
                          "|l_reg=", float(l_reg),
                          "|loss=", float(loss))

                grads = tape.gradient(target=loss, sources=self.modeli_vars_)
                optimizer.apply_gradients(
                    grads_and_vars=zip(grads, self.modeli_vars_))

        print("Training complete. ")
        self.ckpt_.save(self.model_meta_path_ + "/dnn_on_latent_space")
        print("Saved model parameters.")
