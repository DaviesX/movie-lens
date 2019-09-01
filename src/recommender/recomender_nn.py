import tensorflow as tf
import numpy as np
import math as m

from hparams import MOVIE_EMBEDDINGS_SIZE
from hparams import USER_EMBEDDINGS_SIZE

def label():
    with tf.name_scope("label"):
        return tf.placeholder(dtype=tf.float32, 
                              shape=[None], 
                              name="rating")

def embeddings_input(embedding_name: str, embed_size: int):
    with tf.name_scope(embedding_name):
        return tf.placeholder(dtype=tf.float32, 
                              shape=[None, embed_size], 
                              name="input")

def nn_dense_layer(name: str, input, input_size: int, output_size: int):
    with tf.name_scope(name):
        weights = tf.Variable(tf.truncated_normal(shape=[input_size, output_size],
                                                  stddev=1.0/m.sqrt(float(input_size))),
                              name="w")
        biases = tf.Variable(tf.zeros(shape=[output_size]), name="b")
        return tf.nn.relu(tf.matmul(input, weights) + biases, name="layer_output")

def prediction(input, input_size: int):
    return nn_dense_layer(name="rating_pred", 
                          input=input, 
                          input_size=input_size, 
                          output_size=1)

def mse_loss(prediction, label):
    with tf.name_scope("loss"):
        loss = tf.losses.mean_squared_error(label, prediction[:,0])
        return tf.identity(input=loss, name="mse_loss")

class recommender_nn:
    def __init__(self, 
                 model_meta_path: str,
                 model_check_point_dir: str,
                 user_embed_size: int,
                 movie_embed_size: int,
                 embedding_transform: bool,
                 reset_and_train: bool,
                 num_iters: int=100000, 
                 batch_size: int=500,
                 learning_rate: float=0.0001):
        """Construct a NN recommender model.
        
        Arguments:
            model_meta_path {str} -- Location where the model parameters
                are stored/going to be stored.
            model_check_point_dir {str} -- Location where the model check 
                point is stored/goging to be stored.
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
            learning_rate {float} -- The velocity of each 
                gradient descent step. (default: {0.001})
        """
        # Constant hyper-params.
        self.MOVIE_EMBEDDINGS_TRANSFORMED_SIZE = 10
        self.USER_EMBEDDINGS_TRANSFORMED_SIZE = 15
        
        # Configurable hyper-params
        self.user_embed_size_ = user_embed_size
        self.movie_embed_size_ = movie_embed_size
        self.model_meta_path_ = model_meta_path
        self.model_check_point_dir_ = model_check_point_dir
        self.reset_and_train_ = reset_and_train
        self.num_iters_ = num_iters
        self.batch_size_ = batch_size

        self.build_graph(user_embed_size=user_embed_size,
                         movie_embed_size=movie_embed_size,
                         embedding_transform=embedding_transform,
                         learning_rate=learning_rate)

    def build_graph(self, 
                    user_embed_size: int,
                    movie_embed_size: int,
                    embedding_transform: bool,
                    learning_rate: float):
        """Build an NN graph.
        """
        user_embeddings = embeddings_input(embedding_name="user_embeddings", 
                                           embed_size=user_embed_size)
        movie_embeddings = embeddings_input(embedding_name="movie_embeddings", 
                                            embed_size=movie_embed_size)

        if embedding_transform:
            user_embeddings = nn_dense_layer(name="user_embed_transform",
                                             input=user_embeddings,
                                             input_size=user_embed_size,
                                             output_size=self.USER_EMBEDDINGS_TRANSFORMED_SIZE)
            movie_embeddings = nn_dense_layer(name="movie_embed_transform",
                                              input=movie_embeddings,
                                              input_size=movie_embed_size,
                                              output_size=self.MOVIE_EMBEDDINGS_TRANSFORMED_SIZE)
            user_embed_size = self.USER_EMBEDDINGS_TRANSFORMED_SIZE
            movie_embed_size = self.MOVIE_EMBEDDINGS_TRANSFORMED_SIZE

        concat_features = tf.concat(values=[user_embeddings, movie_embeddings],
                                    axis=1, name="concat_features")
        compress_size: int = (self.USER_EMBEDDINGS_TRANSFORMED_SIZE + self.MOVIE_EMBEDDINGS_TRANSFORMED_SIZE)//2
        compress = nn_dense_layer(name="compress", 
                                  input=concat_features, 
                                  input_size=user_embed_size + movie_embed_size,
                                  output_size=compress_size)
        rating_pred = prediction(input=compress, input_size=compress_size)
        
        rating_label = label()
        loss = mse_loss(rating_pred, rating_label)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer.minimize(loss, global_step=global_step, name="optimizer_node")

    def fit(self, 
            user_embed: np.ndarray, 
            movie_embed: np.ndarray, 
            rating: np.ndarray) -> None:
        """Fit the NN model to the user-movie embedding pairs and ratings.
        
        Arguments:
            user_embed {np.ndarray} -- U*E_user matrix where U is
                the number of users and E_user is the embedding size.
            movie_embed {np.ndarray} -- M*E_movie matrix where M is
                the number of movies and E_movie is the embedding size.
            rating {np.ndarray} -- a float array consists of ratings for each
                user-movie pair record.
        Note:
            fit() assumes that user_embed.shape[0] == movie_embed.shape[0] and
            movie_embed.shape[0] == rating.shape[0]
        """
        saver = None
        graph: tf.Graph = None
        
        with tf.Session() as sess:
            if self.reset_and_train_:
                saver = tf.train.Saver()
                graph = tf.get_default_graph()
                sess.run(tf.global_variables_initializer())
            else:
                saver = tf.train.import_meta_graph(
                    meta_graph_or_file=self.model_meta_path_ + ".meta")
                saver.restore(sess, tf.train.latest_checkpoint('./'))
                graph = tf.get_default_graph()

            user_embed_node = graph.get_tensor_by_name("user_embeddings/input:0")
            movie_embed_node = graph.get_tensor_by_name("movie_embeddings/input:0")
            rating_node = graph.get_tensor_by_name("label/rating:0")
            optimizer_node = graph.get_operation_by_name("optimizer_node")
            loss_node = graph.get_tensor_by_name("loss/mse_loss:0")

            dataset_size = user_embed.shape[0]

            for i in range(self.num_iters_):
                batch_idx = np.random.randint(low=0, high=dataset_size,
                                              size=self.batch_size_)
                batch_user_embed = user_embed[batch_idx,:]
                batch_movie_embed = movie_embed[batch_idx,:]
                batch_rating = rating[batch_idx]

                optimizer_node.run(feed_dict={
                    user_embed_node: batch_user_embed,
                    movie_embed_node: batch_movie_embed,
                    rating_node: batch_rating,
                })

                if i % 1000 == 0:
                    loss_val = loss_node.eval(feed_dict={
                        user_embed_node: batch_user_embed,
                        movie_embed_node: batch_movie_embed,
                        rating_node: batch_rating,
                    })
                    print("Loss at ", i, "=", loss_val)
            
            print("Training complete. ")
            saver.save(sess=sess, save_path=self.model_meta_path_)
            print("Saved model parameters.")

    def predict(self, 
                user_embed: np.ndarray, 
                movie_embed: np.ndarray) -> np.ndarray:
        """Make rating prediction on user-movie pair.
        
        Arguments:
            user_embed {np.ndarray} -- U*E_user matrix where U is
                    the number of users and E_user is the embedding size.
            movie_embed {np.ndarray} -- M*E_movie matrix where M is
                    the number of movies and E_movie is the embedding size.
        Note:
            predict() assumes that user_embed.shape[0] == movie_embed.shape[0]
        Returns:
            np.ndarray -- a float array consists N ratings prediction 
                over the user-movie pairs.
        """
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(
                    meta_graph_or_file=self.model_meta_path_ + ".meta")
            saver.restore(sess, tf.train.latest_checkpoint(self.model_check_point_dir_))
            graph = tf.get_default_graph()
            
            user_embed_node = graph.get_tensor_by_name("user_embeddings/input:0")
            movie_embed_node = graph.get_tensor_by_name("movie_embeddings/input:0")
            rating_pred_node = graph.get_tensor_by_name("rating_pred/layer_output:0")

            rating_pred = rating_pred_node.eval(feed_dict={
                user_embed_node: user_embed,
                movie_embed_node: movie_embed,
            })

            return rating_pred


if __name__ == "__main__":
    """Train model using the rated embeddings training set.
    """
    dataset_train = np.load(file="../data/rated_embeddings_train.npy")
    user_embed_train = dataset_train[:, 2:2 + USER_EMBEDDINGS_SIZE]
    movie_embed_train = dataset_train[:, 2 + USER_EMBEDDINGS_SIZE:2 + USER_EMBEDDINGS_SIZE + MOVIE_EMBEDDINGS_SIZE]
    rating_train = dataset_train[:,-1]

    dataset_valid = np.load(file="../data/rated_embeddings_valid.npy")
    user_embed_valid = dataset_valid[:, 2:2 + USER_EMBEDDINGS_SIZE]
    movie_embed_valid = dataset_valid[:, 2 + USER_EMBEDDINGS_SIZE:2 + USER_EMBEDDINGS_SIZE + MOVIE_EMBEDDINGS_SIZE]
    rating_valid = dataset_valid[:,-1]

    model = recommender_nn(model_meta_path="../meta/recommender_nn_transformed.ckpt",
                           model_check_point_dir="../meta",
                           user_embed_size=USER_EMBEDDINGS_SIZE,
                           movie_embed_size=MOVIE_EMBEDDINGS_SIZE,
                           embedding_transform=True,
                           reset_and_train=True)
    model.fit(user_embed=user_embed_train,
              movie_embed=movie_embed_train,
              rating=rating_train)
    
    pred_ratings_train = model.predict(user_embed=user_embed_train,
                                       movie_embed=movie_embed_train)
    pred_ratings_valid = model.predict(user_embed=user_embed_valid,
                                       movie_embed=movie_embed_valid)

    np.save(file="../data/rating_pred_train_nn.npy", arr=pred_ratings_train)
    np.save(file="../data/rating_pred_valid_nn.npy", arr=pred_ratings_valid)