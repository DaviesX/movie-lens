from typing import Tuple

import numpy as np
import tensorflow as tf
import math as m

from hparams import MOVIE_EMBEDDINGS_SIZE
from hparams import USER_EMBEDDINGS_SIZE

def label():
    with tf.name_scope("label"):
        return tf.placeholder(dtype=tf.float32,
                              shape=[None],
                              name="rating")

def movie_id_one_hot_input(num_movies: int):
    with tf.name_scope("movie_input"):
        movie_ids = tf.placeholder(dtype=tf.int32,
                                   shape=[None],
                                   name="movie_id")
        return tf.one_hot(indices=movie_ids, depth=num_movies, name="to_one_hot")

def user_id_one_hot_input(num_users: int):
    with tf.name_scope("user_input"):
        user_ids = tf.placeholder(dtype=tf.int32,
                                  shape=[None],
                                  name="user_id")
        return tf.one_hot(indices=user_ids, depth=num_users, name="to_one_hot")

def embeddings_table(name: str,
                     num_vecs: int,
                     latent_size: int):
    with tf.name_scope(name):
        embed_table = tf.Variable(tf.truncated_normal(shape=[num_vecs, latent_size],
                                                      stddev=1.0/m.sqrt(float(latent_size))),
                                  name="embed_table")
        return embed_table

def nn_dense_layer(name: str,
                   inputs: tf.NodeDef,
                   input_size: int,
                   output_size: int,
                   act_func="relu"):
    with tf.name_scope(name):
        weights = tf.Variable(tf.truncated_normal(shape=[input_size, output_size],
                                                  stddev=1.0/m.sqrt(float(input_size))),
                              name="weights")
        biases = tf.Variable(tf.zeros(shape=[output_size]), name="biases")
        lin_t = tf.matmul(a=inputs, b=weights, name="linear_trans")
        features = tf.add(x=lin_t, y=biases, name="translate")
        if act_func == "relu":
            return tf.nn.relu(features=features, name="layer_output")
        else:
            return tf.tanh(x=features, name="layer_output")

def prediction(inputs: tf.NodeDef, input_size: int):
    preds = nn_dense_layer(name="prediction",
                           inputs=inputs,
                           input_size=input_size,
                           output_size=1)
    return tf.reshape(tensor=preds,
                      shape=[tf.shape(input=preds, name="pred_shape")[0]],
                      name="rating_preds")

def mse_loss(preds: tf.NodeDef, labels: tf.NodeDef):
    with tf.name_scope("loss"):
        loss = tf.losses.mean_squared_error(labels, preds)
        return tf.identity(input=loss, name="mse_loss")

class latent_dnn:
    def __init__(self,
                 model_meta_path: str,
                 model_check_point_dir: str,
                 num_users: int,
                 user_embed_size: int,
                 num_movies: int,
                 movie_embed_size: int,
                 reset_and_train: bool,
                 num_iters=10000,
                 batch_size=200,
                 learning_rate=0.0001):
        """Construct latent spaces for both user and movie by using rating
        prediction as the training task.

        Arguments:
            model_meta_path {str} -- Location where the model parameters
                are stored/going to be stored.
            model_check_point_dir {str} -- Location where the model check-point
                is stored/goging to be stored.
            num_users {int} -- The total number of unique users.
            num_movies {int} -- The total number of unique movies.
            user_embed_size {int} -- The size of user embeddings vector to
                be generated.
            movie_embed_size {int} -- The size of movie embeddings vector
                to be generated.
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
        self.model_check_point_dir_ = model_check_point_dir
        self.reset_and_train_ = reset_and_train
        self.num_iters_ = num_iters
        self.batch_size_ = batch_size

        self.build_graph(num_users=num_users,
                         user_embed_size=user_embed_size,
                         num_movies=num_movies,
                         movie_embed_size=movie_embed_size,
                         learning_rate=learning_rate)

    def build_graph(self, 
                    num_users: int,
                    user_embed_size: int,
                    num_movies: int,
                    movie_embed_size: int,
                    learning_rate: float):
        """Build an NN graph.
        """
        user_ids = user_id_one_hot_input(num_users=num_users)
        movie_ids = movie_id_one_hot_input(num_movies=num_movies)

        user_embed_table = embeddings_table(name="user_embed",
                                            num_vecs=num_users,
                                            latent_size=user_embed_size)
        movie_embed_table = embeddings_table(name="movie_embed",
                                             num_vecs=num_movies,
                                             latent_size=movie_embed_size)

        user_embeddings = tf.matmul(a=user_ids, b=user_embed_table, name="pick_up_user_embed")
        movie_embeddings = tf.matmul(a=movie_ids, b=movie_embed_table, name="pick_up_movie_embed")

        concat_features = tf.concat(values=[user_embeddings, movie_embeddings],
                                    axis=1, name="concat_features")
        compress_size = (user_embed_size + movie_embed_size)//2
        compress = nn_dense_layer(name="compress",
                                  inputs=concat_features,
                                  input_size=user_embed_size + movie_embed_size,
                                  output_size=compress_size)
        rating_preds = prediction(inputs=compress, input_size=compress_size)

        rating_label = label()
        loss = mse_loss(rating_preds, rating_label)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer.minimize(loss, global_step=global_step, name="optimizer_node")

    def fit(self, 
            user_ids: np.ndarray, 
            movie_ids: np.ndarray, 
            rating: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        with tf.Session() as sess:
            saver = None
            if self.reset_and_train_:
                saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
            else:
                saver = tf.train.import_meta_graph(
                    meta_graph_or_file=self.model_meta_path_ + ".meta")
                saver.restore(sess, tf.train.latest_checkpoint('./'))

            graph = tf.get_default_graph()

            user_input_node = graph.get_tensor_by_name("user_input/user_id:0")
            movie_input_node = graph.get_tensor_by_name("movie_input/movie_id:0")
            rating_node = graph.get_tensor_by_name("label/rating:0")
            optimizer_node = graph.get_operation_by_name("optimizer_node")
            loss_node = graph.get_tensor_by_name("loss/mse_loss:0")

            dataset_size = user_ids.shape[0]

            for i in range(self.num_iters_):
                batch_idx = np.random.randint(low=0, high=dataset_size,
                                              size=self.batch_size_)
                batch_user_ids = user_ids[batch_idx]
                batch_movie_ids = movie_ids[batch_idx]
                batch_rating = rating[batch_idx]

                optimizer_node.run(feed_dict={
                    user_input_node: batch_user_ids,
                    movie_input_node: batch_movie_ids,
                    rating_node: batch_rating,
                })

                if i % 100 == 0:
                    loss_val = loss_node.eval(feed_dict={
                        user_input_node: batch_user_ids,
                        movie_input_node: batch_movie_ids,
                        rating_node: batch_rating,
                    })
                    print("Loss at ", i, "=", loss_val)

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
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, self.model_meta_path_)

            graph = tf.get_default_graph()

            user_input_node = graph.get_tensor_by_name("user_input/user_id:0")
            movie_input_node = graph.get_tensor_by_name("movie_input/movie_id:0")
            rating_preds_node = graph.get_tensor_by_name("rating_preds:0")

            dataset_size = user_ids.shape[0]
            rating_pred = np.zeros(shape=(dataset_size))
            for i in range(0, dataset_size, self.batch_size_):
                pred = rating_preds_node.eval(feed_dict={
                    user_input_node: user_ids[i:i + self.batch_size_],
                    movie_input_node: movie_ids[i:i + self.batch_size_],
                })
                rating_pred[i:i + pred.shape[0]] = pred

            return rating_pred


if __name__ == "__main__":
    """Find latent space for both users and movies.
    """
    dataset_train = np.load(file="../data/rated_embeddings_train.npy")
    user_ids = dataset_train[:, 0].astype(dtype=np.int32) - 1
    movie_ids = dataset_train[:, 1].astype(dtype=np.int32) - 1
    rating_train = dataset_train[:, -1]

    dataset_valid = np.load(file="../data/rated_embeddings_valid.npy")
    user_ids_valid = dataset_valid[:, 0]
    movie_ids_valid = dataset_valid[:, 1]

    num_users = np.max(user_ids) + 1
    num_movies = np.max(movie_ids) + 1

    model = latent_dnn(model_meta_path="../meta/latent_dnn.ckpt",
                       model_check_point_dir="../meta",
                       num_users=num_users,
                       user_embed_size=USER_EMBEDDINGS_SIZE,
                       num_movies=num_movies,
                       movie_embed_size=MOVIE_EMBEDDINGS_SIZE,
                       num_iters=10000,
                       reset_and_train=True)
    # model.fit(user_ids=user_ids,
    #           movie_ids=movie_ids,
    #           rating=rating_train)

    user_embed, movie_embed = model.export_embeddings()
    
    np.save(file="../data/embeddings_user_train.npy", arr=user_embed)
    np.save(file="../data/embeddings_user_valid.npy", arr=user_embed)

    np.save(file="../data/embeddings_movie_train.npy", arr=movie_embed)
    np.save(file="../data/embeddings_movie_valid.npy", arr=movie_embed)

    pred_ratings_train = model.predict_ratings(user_ids=user_ids,
                                               movie_ids=movie_ids)
    pred_ratings_valid = model.predict_ratings(user_ids=user_ids_valid,
                                               movie_ids=movie_ids_valid)

    np.save(file="../data/rating_pred_train_latent_nn.npy", arr=pred_ratings_train)
    np.save(file="../data/rating_pred_valid_latent_nn.npy", arr=pred_ratings_valid)
