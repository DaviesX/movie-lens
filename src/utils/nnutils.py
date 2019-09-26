from typing import List, Dict
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
            initial_value=tf.random.truncated_normal(shape=(output_size, input_size)))
        self.biases_ = tf.Variable(initial_value=tf.zeros(shape=(output_size)))

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
        t = self.weights_ * x + self.biases_
        if act_fn == "tanh":
            return tf.tanh(t)
        elif act_fn == "relu":
            return tf.nn.leaky_relu(t)
        else:
            return t

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
