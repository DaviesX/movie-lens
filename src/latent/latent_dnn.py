from typing import Tuple, List

import numpy as np
import tensorflow as tf
import math as m

from hparams import MOVIE_EMBEDDINGS_SIZE
from hparams import USER_EMBEDDINGS_SIZE

def label() -> tf.Tensor:
    with tf.name_scope("label"):
        return tf.placeholder(dtype=tf.float32,
                              shape=[None],
                              name="rating")

def movie_id_one_hot_input(num_movies: int) -> tf.Tensor:
    with tf.name_scope("movie_input"):
        movie_ids = tf.placeholder(dtype=tf.int32,
                                   shape=[None],
                                   name="movie_id")
        return tf.one_hot(indices=movie_ids, depth=num_movies, name="to_one_hot")

def user_id_one_hot_input(num_users: int) -> tf.Tensor:
    with tf.name_scope("user_input"):
        user_ids = tf.placeholder(dtype=tf.int32,
                                  shape=[None],
                                  name="user_id")
        return tf.one_hot(indices=user_ids, depth=num_users, name="to_one_hot")

def embeddings_table(name: str,
                     num_vecs: int,
                     latent_size: int,
                     vars: list) -> tf.Tensor:
    with tf.name_scope(name):
        embed_table = tf.Variable(tf.truncated_normal(shape=[num_vecs, latent_size],
                                                      stddev=1.0/m.sqrt(float(latent_size))),
                                  name="embed_table")
        vars.append(embed_table)
        return embed_table

def keep_prob(name: str):
    with tf.name_scope(name):
        return tf.placeholder(tf.float32, name="keep_prob")

def nn_dense_layer(name: str,
                   inputs: tf.Tensor,
                   input_size: int,
                   output_size: int,
                   act_func: str,
                   vars: List[tf.Tensor],
                   reg_vars: List[tf.Tensor]) -> tf.Tensor:
    with tf.name_scope(name):
        weights = tf.Variable(tf.truncated_normal(shape=[input_size, output_size],
                                                  stddev=1.0/m.sqrt(float(input_size))),
                              name="weights")
        biases = tf.Variable(tf.zeros(shape=[output_size]), name="biases")

        reg_vars.append(weights)
        vars.append(weights)
        vars.append(biases)

        lin_t = tf.matmul(a=inputs, b=weights, name="linear_trans")
        features = tf.add(x=lin_t, y=biases, name="translate")
        if act_func == "relu":
            return tf.nn.relu(features=features, name="layer_output")
        elif act_func == "tanh":
            return tf.tanh(x=features, name="layer_output")
        else:
            raise "Unknown activaion function " + act_func

def prediction(inputs: tf.Tensor,
               input_size: int,
               vars: List[tf.Tensor],
               reg_vars: List[tf.Tensor]) -> tf.Tensor:
    preds = nn_dense_layer(name="prediction",
                           inputs=inputs,
                           input_size=input_size,
                           output_size=1,
                           act_func="tanh",
                           vars=vars,
                           reg_vars=reg_vars)
    preds = 2.5*(preds + 1)
    preds = tf.reshape(tensor=preds,
                       shape=[tf.shape(input=preds, name="pred_shape")[0]],
                       name="rating_preds")
    return preds

def rating_pred_loss(preds: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    with tf.name_scope("rating_pred_loss"):
        loss = tf.losses.mean_squared_error(labels, preds)
        return tf.identity(input=loss, name="mse")

def rating_regularizer(weights: List[tf.Tensor]) -> tf.Tensor:
    with tf.name_scope("rating_regularizer"):
        loss = None
        for w in weights:
            if loss is None:
                loss = tf.reduce_mean(input_tensor=w*w)
            else:
                loss += tf.reduce_mean(input_tensor=w*w)
        return loss

def embeddings_unit_norm_loss(name: str, embeddings: tf.Tensor) -> tf.Tensor:
    with tf.name_scope(name):
        norm = tf.sqrt(x=tf.reduce_sum(input_tensor=embeddings*embeddings,
                                       axis=1),
                       name="embedding_norm")
        len_diffs = 1.0 - norm
        errs = len_diffs*len_diffs
        return tf.reduce_mean(input_tensor=errs, name="err")

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

        self.build_graph(num_users=num_users,
                         user_embed_size=user_embed_size,
                         num_movies=num_movies,
                         movie_embed_size=movie_embed_size,
                         indirect_cause=indirect_cause,
                         learning_rate=learning_rate)

    def build_graph(self,
                    num_users: int,
                    user_embed_size: int,
                    num_movies: int,
                    movie_embed_size: int,
                    indirect_cause: bool,
                    learning_rate: float):
        """Build a multi-task embeddings model.
        """
        tf.reset_default_graph()

        embeddings_vars = list()
        rating_pred_vars = list()
        reg_vars = list()

        user_ids = user_id_one_hot_input(num_users=num_users)
        movie_ids = movie_id_one_hot_input(num_movies=num_movies)

        user_embed_table = embeddings_table(name="user_embed",
                                            num_vecs=num_users,
                                            latent_size=user_embed_size,
                                            vars=embeddings_vars)
        movie_embed_table = embeddings_table(name="movie_embed",
                                             num_vecs=num_movies,
                                             latent_size=movie_embed_size,
                                             vars=embeddings_vars)

        user_embeddings = tf.matmul(a=user_ids, b=user_embed_table, name="pick_up_user_embed")
        movie_embeddings = tf.matmul(a=movie_ids, b=movie_embed_table, name="pick_up_movie_embed")

        concat_features = tf.concat(values=[user_embeddings, movie_embeddings],
                                    axis=1, name="concat_features")
        concat_features = tf.nn.dropout(x=concat_features, keep_prob=keep_prob(name="concat"))

        if indirect_cause:
            compress_size = (user_embed_size + movie_embed_size)//2
            features = nn_dense_layer(name="compress",
                                      inputs=concat_features,
                                      input_size=user_embed_size + movie_embed_size,
                                      output_size=compress_size,
                                      act_func="tanh",
                                      vars=rating_pred_vars,
                                      reg_vars=reg_vars)
            feature_size = compress_size
        else:
            features = concat_features
            feature_size = user_embed_size + movie_embed_size

        rating_preds = prediction(inputs=features,
                                  input_size=feature_size,
                                  vars=rating_pred_vars,
                                  reg_vars=reg_vars)

        rating_label = label()
        pred_loss = rating_pred_loss(preds=rating_preds,
                                     labels=rating_label)
        rating_loss = tf.identity(input= pred_loss + \
                                         rating_regularizer(weights=reg_vars),
                                  name="rating_loss")
        embeddings_loss = tf.identity(input=pred_loss, name="embeddings_loss")

        optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer.minimize(loss=embeddings_loss,
                           var_list=embeddings_vars,
                           name="embeddings_task")
        optimizer.minimize(loss=rating_loss,
                           var_list=rating_pred_vars,
                           name="rating_pred_task")

    def fit(self,
            user_embed_table: np.ndarray,
            movie_embed_table: np.ndarray,
            user_ids: np.ndarray,
            movie_ids: np.ndarray,
            rating: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        user_embed_table = user_embed_table.astype(dtype=np.float32)
        movie_embed_table = movie_embed_table.astype(dtype=np.float32)
        user_ids = user_ids.astype(dtype=np.int32)
        movie_ids = movie_ids.astype(dtype=np.int32)
        rating = rating.astype(dtype=np.float32)

        with tf.Session() as sess:
            saver = tf.train.Saver()
            if self.reset_and_train_:
                sess.run(tf.global_variables_initializer())
            else:
                saver.restore(sess, self.model_meta_path_)

            graph = tf.get_default_graph()

            if user_embed_table is not None:
                table = graph.get_tensor_by_name("user_embed/embed_table:0")
                sess.run(tf.assign(ref=table, value=user_embed_table))

            if movie_embed_table is not None:
                table = graph.get_tensor_by_name("movie_embed/embed_table:0")
                sess.run(tf.assign(ref=table, value=movie_embed_table))

            user_input_node = graph.get_tensor_by_name("user_input/user_id:0")
            movie_input_node = graph.get_tensor_by_name("movie_input/movie_id:0")
            rating_node = graph.get_tensor_by_name("label/rating:0")
            concat_keep_prob = graph.get_tensor_by_name("concat/keep_prob:0")

            embeddings_task = graph.get_operation_by_name("embeddings_task")
            rating_pred_task = graph.get_operation_by_name("rating_pred_task")

            rating_loss_node = graph.get_tensor_by_name("rating_loss:0")
            embeddings_loss_node = graph.get_tensor_by_name("embeddings_loss:0")

            dataset_size = user_ids.shape[0]

            curr_task = "rating_pred"
            num_iters_for_curr_task = 0
            for i in range(self.num_iters_):
                batch_idx = np.random.randint(low=0, high=dataset_size,
                                              size=self.batch_size_)
                batch_user_ids = user_ids[batch_idx]
                batch_movie_ids = movie_ids[batch_idx]
                batch_rating = rating[batch_idx]

                if num_iters_for_curr_task > 1000 and curr_task == "embeddings":
                    curr_task = "rating_pred"
                    num_iters_for_curr_task = 0
                if num_iters_for_curr_task > 1000 and curr_task == "rating_pred":
                    curr_task = "embeddings"
                    num_iters_for_curr_task = 0

                if curr_task == "embeddings":
                    embeddings_task.run(feed_dict={
                        user_input_node: batch_user_ids,
                        movie_input_node: batch_movie_ids,
                        rating_node: batch_rating,
                        concat_keep_prob: 0.7
                    })
                elif curr_task == "rating_pred":
                    rating_pred_task.run(feed_dict={
                        user_input_node: batch_user_ids,
                        movie_input_node: batch_movie_ids,
                        rating_node: batch_rating,
                        concat_keep_prob: 0.7
                    })
                num_iters_for_curr_task += 1

                if i % 100 == 0:
                    losses = sess.run(fetches=[embeddings_loss_node,
                                               rating_loss_node],
                                      feed_dict={
                                          user_input_node: batch_user_ids,
                                          movie_input_node: batch_movie_ids,
                                          rating_node: batch_rating,
                                          concat_keep_prob: 1.0
                                      })
                    print("At", i,
                          "|task=", curr_task,
                          "|embeddings_loss=", losses[0],
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
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, self.model_meta_path_)

            graph = tf.get_default_graph()

            user_embed_table_node = graph.get_tensor_by_name("user_embed/embed_table:0")
            movie_embed_table_node = graph.get_tensor_by_name("movie_embed/embed_table:0")

            return user_embed_table_node.eval(), movie_embed_table_node.eval()


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

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, self.model_meta_path_)

            graph = tf.get_default_graph()

            user_input_node = graph.get_tensor_by_name("user_input/user_id:0")
            movie_input_node = graph.get_tensor_by_name("movie_input/movie_id:0")
            rating_preds_node = graph.get_tensor_by_name("rating_preds:0")
            concat_keep_prob = graph.get_tensor_by_name("concat/keep_prob:0")

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
