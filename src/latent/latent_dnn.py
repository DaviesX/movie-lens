from typing import Tuple, List

import numpy as np
import tensorflow as tf
import math as m

from utils import nnutils
from hparams import MOVIE_EMBEDDINGS_SIZE
from hparams import USER_EMBEDDINGS_SIZE


@tf.function
def rating_prediction_loss(preds: np.ndarray, labels: np.ndarray) -> float:
    """[summary]

    Arguments:
        preds {np.ndarray} -- [description]
        labels {np.ndarray} -- [description]

    Returns:
        np.ndarray -- [description]
    """
    return tf.metrics.mean_squared_error(labels, preds)


@tf.function
def regularizer_loss(weights: List[tf.Tensor], alpha=0.04) -> float:
    """[summary]

    Arguments:
        weights {List[tf.Tensor]} -- [description]

    Keyword Arguments:
        alpha {float} -- [description] (default: {0.04})

    Returns:
        float -- [description]
    """
    regularizer = tf.keras.regularizers.l2(l=alpha)
    loss = 0
    for weight in weights:
        loss += regularizer(weight)
    return loss


class latent_dnn:
    def __init__(self,
                 model_meta_path: str,
                 num_users: int,
                 user_embed_size: int,
                 num_movies: int,
                 movie_embed_size: int,
                 indirect_cause: bool,
                 reset_and_train: bool,
                 num_iters=10000,
                 batch_size=200,
                 learning_rate=0.001):
        """Construct latent spaces for both user and movie by using rating
        prediction as the training task.

        Arguments:
            model_meta_path {str} -- Location where the model parameters
                are stored/going to be stored.
            num_users {int} -- The total number of unique users.
            num_movies {int} -- The total number of unique movies.
            user_embed_size {int} -- The size of user embeddings vector to
                be generated.
            movie_embed_size {int} -- The size of movie embeddings vector
                to be generated.
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

        self.num_users_ = num_users
        self.num_movies_ = num_movies

        # Model varibles
        self.user_embed_table_ = tf.Variable(initial_value=tf.ones(shape=()))
        self.movie_embed_table_ = None

    @tf.function
    def predict_ratings(self,
                        user_embed: tf.Tensor,
                        movie_embed: tf.Tensor,
                        user_embed_size: int,
                        movie_embed_size: int,
                        indirect_cause: bool,
                        drop_prob: float,
                        pred_vars: List[tf.Tensor],
                        reg_vars: List[tf.Tensor]) -> np.ndarray:
        """[summary]

        Arguments:
            user_embed {tf.Tensor} -- [description]
            movie_embed {tf.Tensor} -- [description]
            user_embed_size {int} -- [description]
            movie_embed_size {int} -- [description]
            indirect_cause {bool} -- [description]
            drop_prob {float} -- [description]
            pred_vars {List[tf.Tensor]} -- [description]
            reg_vars {List[tf.Tensor]} -- [description]

        Returns:
            np.ndarray -- [description]
        """
        concat_features = tf.concat(values=[user_embed, movie_embed],
                                    axis=1, name="concat_features")
        concat_features = tf.nn.dropout(x=concat_features, rate=drop_prob)

        if indirect_cause:
            compress_size = (user_embed_size + movie_embed_size)//2
            features = transform_inputs(inputs=concat_features,
                                        input_size=user_embed_size + movie_embed_size,
                                        output_size=compress_size,
                                        act_func="tanh",
                                        all_vars=pred_vars,
                                        reg_vars=reg_vars)
            feature_size = compress_size
        else:
            features = concat_features
            feature_size = user_embed_size + movie_embed_size

        transform = nnutils.transform(input_size=features, output_size=1)
        preds = 2.5*(transform(features) + 1)
        return preds[:, 0]

    def build_graph(self,
                    num_users: int,
                    user_embed_size: int,
                    num_movies: int,
                    movie_embed_size: int,
                    indirect_cause: bool,
                    learning_rate: float):
        """Build a multi-task embeddings model.
        """
        self.user_embed_table_ = tf.Variable(
            shape=[num_users, user_embed_size])
        self.movie_embed_table_ = tf.Variable(
            shape=[num_movies, movie_embed_size])

        with tf.name_scope("rating_pred_group"):
            self.build_rating_pred_task(
                num_users=num_users,
                num_movies=num_movies,
                user_embed_table=self.user_embed_table_,
                movie_embed_table=self.movie_embed_table_,
                user_embed_size=user_embed_size,
                movie_embed_size=movie_embed_size,
                indirect_cause=indirect_cause,
                learning_rate=learning_rate)

    def build_rating_pred_task(self,
                               num_users: int,
                               num_movies: int,
                               user_embed_table: tf.Tensor,
                               movie_embed_table: tf.Tensor,
                               user_embed_size: int,
                               movie_embed_size: int,
                               indirect_cause: bool,
                               learning_rate: float):
        """Train embedding vectors by making rating predictions given the pair
        (USER_ID, MOVIE_ID).
        """
        user_ids = user_id_one_hot_input(
            name="user_input", num_users=num_users)
        movie_ids = movie_id_one_hot_input(
            name="movie_input", num_movies=num_movies)

        user_embed = tf.matmul(a=user_ids,
                               b=user_embed_table,
                               name="pick_up_user_embed")
        movie_embed = tf.matmul(a=movie_ids,
                                b=movie_embed_table,
                                name="pick_up_movie_embed")

        rating_pred_vars = list()
        reg_vars = list()

        rating_preds = self.build_predict_rating(user_embed=user_embed,
                                                 movie_embed=movie_embed,
                                                 user_embed_size=user_embed_size,
                                                 movie_embed_size=movie_embed_size,
                                                 indirect_cause=indirect_cause,
                                                 pred_vars=rating_pred_vars,
                                                 reg_vars=reg_vars)

        pred_loss = rating_pred_loss(preds=rating_preds,
                                     labels=rating_label())
        rating_loss = tf.identity(input=pred_loss +
                                  rating_regularizer(weights=reg_vars),
                                  name="rating_loss")
        embeddings_loss = tf.identity(input=pred_loss,
                                      name="rating_embeddings_loss")

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        optimizer.minimize(loss=embeddings_loss,
                           var_list=[user_embed_table, movie_embed_table],
                           name="rating_embeddings_task")
        optimizer.minimize(loss=rating_loss,
                           var_list=rating_pred_vars,
                           name="rating_pred_task")

    def sample_users_and_movies(self,
                                user_ids: np.ndarray,
                                movie_ids: np.ndarray,
                                ratings: np.ndarray,
                                batch_size: int) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dataset_size = ratings.shape[0]
        batch_idx = np.random.randint(low=0, high=dataset_size,
                                      size=batch_size)
        batch_user_ids = user_ids[batch_idx]
        batch_movie_ids = movie_ids[batch_idx]
        batch_ratings = ratings[batch_idx]

        return batch_user_ids, batch_movie_ids, batch_ratings

    def fit(self,
            user_embed_table: np.ndarray,
            movie_embed_table: np.ndarray,
            user_ids: np.ndarray,
            movie_ids: np.ndarray,
            ratings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the NN model to construct latent spaces for both user and movie
        by using rating prediction as the training task.

        Arguments:
            user_embed_table {np.ndarray} -- Optional, initial embeddings
                vector for each user.
            movie_embed_table {np.ndarray} -- Optional, initial embeddings
                vector for each movie.
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
        user_ids = user_ids.astype(dtype=np.int32)
        movie_ids = movie_ids.astype(dtype=np.int32)
        ratings = ratings.astype(dtype=np.float32)

        with tf.compat.v1.Session() as sess:
            saver = tf.compat.v1.train.Saver()
            if self.reset_and_train_:
                sess.run(tf.compat.v1.global_variables_initializer())
            else:
                saver.restore(sess, self.model_meta_path_)

            graph = tf.compat.v1.get_default_graph()
            """
            if user_embed_table is not None:
                table = graph.get_tensor_by_name("user_embed/embed_table:0")
                sess.run(tf.compat.v1.assign(ref=table,
                                   value=user_embed_table.astype(dtype=np.float32)))

            if movie_embed_table is not None:
                table = graph.get_tensor_by_name("movie_embed/embed_table:0")
                sess.run(tf.compat.v1.assign(ref=table,
                                   value=movie_embed_table.astype(dtype=np.float32)))
            """
            user_input_node = graph.get_tensor_by_name(
                "rating_pred_group/user_input/user_id:0")
            movie_input_node = graph.get_tensor_by_name(
                "rating_pred_group/movie_input/movie_id:0")
            rating_node = graph.get_tensor_by_name(
                "rating_pred_group/label/rating:0")
            concat_keep_prob = graph.get_tensor_by_name(
                "rating_pred_group/concat/keep_prob:0")

            rating_embed_task = graph.get_operation_by_name(
                "rating_pred_group/rating_embeddings_task")
            rating_pred_task = graph.get_operation_by_name(
                "rating_pred_group/rating_pred_task")

            rating_loss_node = graph.get_tensor_by_name(
                "rating_pred_group/rating_loss:0")
            rating_embed_loss_node = graph.get_tensor_by_name(
                "rating_pred_group/rating_embeddings_loss:0")

            curr_task = "rating_pred"
            num_iters_for_curr_task = 0
            for i in range(self.num_iters_):
                if num_iters_for_curr_task > 1000 and curr_task == "embed_rating":
                    curr_task = "rating_pred"
                    num_iters_for_curr_task = 0
                if num_iters_for_curr_task > 1000 and curr_task == "rating_pred":
                    curr_task = "embed_rating"
                    num_iters_for_curr_task = 0

                if curr_task == "embed_rating":
                    batch_user_ids, batch_movie_ids, batch_ratings = \
                        self.sample_users_and_movies(user_ids=user_ids,
                                                     movie_ids=movie_ids,
                                                     ratings=ratings,
                                                     batch_size=self.batch_size_)
                    rating_embed_task.run(feed_dict={
                        user_input_node: batch_user_ids,
                        movie_input_node: batch_movie_ids,
                        rating_node: batch_ratings,
                        concat_keep_prob: 0.7
                    })
                elif curr_task == "rating_pred":
                    batch_user_ids, batch_movie_ids, batch_ratings = \
                        self.sample_users_and_movies(user_ids=user_ids,
                                                     movie_ids=movie_ids,
                                                     ratings=ratings,
                                                     batch_size=self.batch_size_)
                    rating_pred_task.run(feed_dict={
                        user_input_node: batch_user_ids,
                        movie_input_node: batch_movie_ids,
                        rating_node: batch_ratings,
                        concat_keep_prob: 0.7
                    })
                num_iters_for_curr_task += 1

                if i % 100 == 0:
                    batch_user_ids, batch_movie_ids, batch_ratings = \
                        self.sample_users_and_movies(user_ids=user_ids,
                                                     movie_ids=movie_ids,
                                                     ratings=ratings,
                                                     batch_size=self.batch_size_)
                    losses = sess.run(fetches=[rating_embed_loss_node,
                                               rating_loss_node],
                                      feed_dict={
                                          user_input_node: batch_user_ids,
                                          movie_input_node: batch_movie_ids,
                                          rating_node: batch_ratings,
                                          concat_keep_prob: 1.0
                    })
                    print("Epoch", round(i*self.batch_size_/ratings.shape[0], 3),
                          "|task=", curr_task,
                          "|rating_embed_loss=", losses[0],
                          "|rating_loss=", losses[1])

            print("Training complete. ")
            saver.save(sess=sess, save_path=self.model_meta_path_)
            print("Saved model parameters.")

    def export_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Export the fitted user and movie embeddings.

        Returns:
            Tuple[np.ndarray, np.ndarray] -- A tuple of user embeddings and
                movie embeddings
        """
        return self.user_embed_table_.eval(), self.movie_embed_table_.eval()

    def predict_ratings(self,
                        user_ids: np.ndarray,
                        movie_ids: np.ndarray) -> np.ndarray:
        """Make rating prediction on user-movie pair.

        Arguments:
            user_ids {np.ndarray} -- A list of user ids pairing up with the
                movie_ids.
            movie_ids {np.ndarray} -- A list of movie ids pairing up with the
                user_ids.
        Note:
            predict() assumes that user_embed.shape[0] == movie_embed.shape[0]
        Returns:
            np.ndarray -- a float array consists N ratings prediction
                over the user-movie pairs.
        """
        user_ids = user_ids.astype(dtype=np.int32)
        movie_ids = movie_ids.astype(dtype=np.int32)

        saver = tf.compat.v1.train.Saver()

        with tf.compat.v1.Session() as sess:
            saver.restore(sess, self.model_meta_path_)

            graph = tf.compat.v1.get_default_graph()

            user_input_node = graph.get_tensor_by_name(
                "rating_pred_group/user_input/user_id:0")
            movie_input_node = graph.get_tensor_by_name(
                "rating_pred_group/movie_input/movie_id:0")
            rating_preds_node = graph.get_tensor_by_name(
                "rating_pred_group/rating_preds:0")
            concat_keep_prob = graph.get_tensor_by_name(
                "rating_pred_group/concat/keep_prob:0")

            dataset_size = user_ids.shape[0]
            rating_pred = np.zeros(shape=(dataset_size))
            for i in range(0, dataset_size, self.batch_size_):
                pred = rating_preds_node.eval(feed_dict={
                    user_input_node: user_ids[i:i + self.batch_size_],
                    movie_input_node: movie_ids[i:i + self.batch_size_],
                    concat_keep_prob: 1.0
                })
                rating_pred[i:i + pred.shape[0]] = pred

            return rating_pred
