import tensorflow as tf

"""
https://stackoverflow.com/questions/47060685/chamfer-distance-between-two-point-clouds-in-tensorflow
"""


def distance_matrix(array1, array2):
    """
    arguments: 
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
            , it's size: (num_point, num_point)
    """
    num_point, num_features = array1.shape
    expanded_array1 = tf.tile(array1, (num_point, 1))
    expanded_array2 = tf.reshape(
        tf.tile(tf.expand_dims(array2, 1),
                (1, num_point, 1)),
        (-1, num_features))
    distances = tf.norm(expanded_array1 - expanded_array2, axis=1)
    distances = tf.reshape(distances, (num_point, num_point))
    return distances


def chamfer_min(array1, array2):
    distances = distance_matrix(array1, array2)
    distances_min = tf.reduce_min(distances, axis=1)
    indices = tf.argmax(distances, axis=1)
    return distances_min, indices
