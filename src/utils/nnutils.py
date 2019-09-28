from typing import List, Dict
from math import sqrt
import numpy as np
import tensorflow as tf


class transform:
    def __init__(self, input_size, output_size):
        """[summary]

        Arguments:
            input_size {[type]} -- [description]
            output_size {[type]} -- [description]
        """
        self.weights_ = tf.Variable(
            initial_value=tf.random.truncated_normal(shape=(output_size, input_size),
                                                     stddev=1/sqrt(float(input_size))))
        self.biases_ = tf.Variable(
            initial_value=tf.zeros(shape=(output_size, 1)))

    @tf.function
    def __call__(self, x: np.ndarray, act_fn="tanh") -> np.ndarray:
        """[summary]

        Arguments:
            x {np.ndarray} -- [description]

        Keyword Arguments:
            act_fn {str} -- [description] (default: {"tanh"})

        Returns:
            np.ndarray -- [description]
        """
        t_x = tf.matmul(self.weights_, x) + self.biases_
        if act_fn == "tanh":
            return tf.tanh(t_x)
        elif act_fn == "relu":
            return tf.nn.leaky_relu(t_x)
        else:
            return t_x

    def weights(self) -> tf.Tensor:
        """[summary]

        Returns:
            tf.Tensor -- [description]
        """
        return self.weights_

    def biases(self) -> tf.Tensor:
        """[summary]

        Returns:
            tf.Tensor -- [description]
        """
        return self.biases_


@tf.function
def regularizer_loss(weights: List[tf.Tensor], alpha=0.01) -> float:
    """[summary]

    Arguments:
        weights {List[tf.Tensor]} -- [description]

    Keyword Arguments:
        alpha {float} -- [description] (default: {0.01})

    Returns:
        float -- [description]
    """
    regularizer = tf.keras.regularizers.l2(l=alpha)
    loss = 0
    for weight in weights:
        loss += regularizer(weight)
    return loss


def collect_transform_vars(transforms: List[transform]) -> Dict[str, tf.Tensor]:
    """[summary]

    Arguments:
        transforms {List[transform]} -- [description]

    Returns:
        Dict[str, tf.Tensor] -- [description]
    """
    var_dict = dict()
    for i in range(len(transforms)):
        var_dict["t_" + str(i) + "_w"] = transforms[i].weights()
        var_dict["t_" + str(i) + "_b"] = transforms[i].biases()
    return var_dict


# a = transform(3, 3)
# b = transform(2, 1)

# ckpt = tf.train.Checkpoint(**collect_transform_vars(transforms=[a, b]))
# # ckpt.save("./test")
# status = ckpt.restore(tf.train.latest_checkpoint("."))
# status.assert_consumed()
