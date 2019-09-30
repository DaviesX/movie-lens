from typing import Tuple, List
import tensorflow as tf
import numpy as np

from utils import nnutils


def sample_batch(user_ids: np.ndarray,
                 movie_ids: np.ndarray,
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
    batch_user_ids = user_ids[batch_idx]
    batch_movie_ids = movie_ids[batch_idx]
    batch_ratings = ratings[batch_idx]

    return batch_user_ids, batch_movie_ids, batch_ratings


class dnn_on_latent_space:
    def __init__(self,
                 model_meta_path: str,
                 user_embed_table: np.ndarray,
                 movie_embed_table: np.ndarray,
                 reset_and_train: bool,
                 num_iters=200000,
                 batch_size=400,
                 learning_rate=0.001):
        """Construct a deep neural network missing-value completion model based on
        a latent space that explains users and movies.

        Arguments:
            model_meta_path {str} -- Location where the model parameters
                are stored/going to be stored.
            user_embed_table {np.ndarray} -- embedding vectors for each user.
            movie_embed_table {np.ndarray} -- embedding vectors for each movie.
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
        self.user_embed_table_ = user_embed_table
        self.movie_embed_table_ = movie_embed_table
        self.model_meta_path_ = model_meta_path
        self.reset_and_train_ = reset_and_train
        self.num_iters_ = num_iters
        self.batch_size_ = batch_size
        self.learning_rate_ = learning_rate

        # Model varibles
        user_embed_size = self.user_embed_table_.shape[1]
        movie_embed_size = self.movie_embed_table_.shape[1]
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
    def predict_(self,
                 user_ids: np.ndarray,
                 movie_ids: np.ndarray,
                 drop_prob=0.0) -> np.ndarray:
        """Predict the rating value of each (user_id, movie_id) pair by using
        their representation vector pair (user_embed, movie_embed).

        Arguments:
            user_ids {np.ndarray} -- A list of user ids pairing up with the
                movie_ids.
            movie_ids {np.ndarray} -- A list of movie ids pairing up with the
                user_ids.

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

        features = self.ratings_t1i_(x=concat_features, act_fn="tanh")
        preds = 2.5 * \
            (self.ratings_t2i_(x=features, act_fn="tanh") + 1)

        return preds[0, :]

    def predict(self,
                user_ids: np.ndarray,
                movie_ids: np.ndarray) -> np.ndarray:
        """Same as predict_() but performed in batch mode.

        Returns:
            np.ndarray -- rating predictions.
        """
        num_records = user_ids.shape[0]
        preds = np.zeros(shape=(num_records))
        for i in range(0, num_records, self.batch_size_):
            begin = i
            end = i + self.batch_size_
            preds[begin:end] = self.predict_(user_ids=user_ids[begin:end],
                                             movie_ids=movie_ids[begin:end],
                                             drop_prob=0.0)
        return preds

    def fit(self,
            user_ids: np.ndarray,
            movie_ids: np.ndarray,
            ratings: np.ndarray) -> None:
        """Fit the NN model to the user-movie embedding pairs and ratings.

        Arguments:
            user_ids {np.ndarray} -- A list of user ids pairing up with the
                movie_ids.
            movie_ids {np.ndarray} -- A list of movie ids pairing up with the
                user_ids.
            ratings {np.ndarray} -- a float array consists of ratings for each
                user-movie pair record.
        Note:
            fit() assumes that user_embed.shape[0] == movie_embed.shape[0] and
            movie_embed.shape[0] == rating.shape[0]
        """
        optimizer = tf.optimizers.Adamax(learning_rate=self.learning_rate_)

        for i in range(self.num_iters_):
            progress = (i + 1.0)/self.num_iters_

            batch_user_ids, batch_movie_ids, batch_ratings = \
                sample_batch(user_ids=user_ids,
                             movie_ids=movie_ids,
                             ratings=ratings,
                             batch_size=self.batch_size_)

            with tf.GradientTape() as tape:
                ratings_hat = self.predict_(
                    user_ids=batch_user_ids, movie_ids=batch_movie_ids, drop_prob=0.3)
                l_ratings = tf.metrics.mean_squared_error(
                    y_true=batch_ratings, y_pred=ratings_hat)
                l_reg = nnutils.regularizer_loss(weights=self.regi_vars_)
                loss = l_ratings + l_reg

                if i % 1000 == 0:
                    print("Epoch", round(i*self.batch_size_/ratings.shape[0], 1),
                          "|p=", int(progress*100), "%",
                          "|l_ratings=", round(float(l_ratings), 2),
                          "|l_reg=", round(float(l_reg), 2),
                          "|loss=", round(float(loss), 2))

                grads = tape.gradient(target=loss, sources=self.modeli_vars_)
                optimizer.apply_gradients(
                    grads_and_vars=zip(grads, self.modeli_vars_))

        print("Training complete. ")
        self.ckpt_.save(self.model_meta_path_ + "/dnn_on_latent_space")
        print("Saved model parameters.")
