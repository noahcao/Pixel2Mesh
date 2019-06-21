"""
Approxmiate algorithm for computing the Earch Mover's Distance.

Original author: Haoqiang Fan
Modified by Charles R. Qi
"""

import tensorflow as tf
from tensorflow.python.framework import ops
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
approxmatch_module = tf.load_op_library(os.path.join(BASE_DIR, 'libtf_approxmatch.so'))


def approx_match(xyz1, xyz2):
    '''
input:
    xyz1 : batch_size * #dataset_points * 3
    xyz2 : batch_size * #query_points * 3
returns:
    match : batch_size * #query_points * #dataset_points
    '''
    return approxmatch_module.approx_match(xyz1, xyz2)


ops.NoGradient('ApproxMatch')


def match_cost(xyz1, xyz2, match):
    '''
input:
    xyz1 : batch_size * #dataset_points * 3
    xyz2 : batch_size * #query_points * 3
    match : batch_size * #query_points * #dataset_points
returns:
    cost : batch_size
    '''
    return approxmatch_module.match_cost(xyz1, xyz2, match)


@tf.RegisterGradient('MatchCost')
def _match_cost_grad(op, grad_cost):
    xyz1 = op.inputs[0]
    xyz2 = op.inputs[1]
    match = op.inputs[2]
    grad_1, grad_2 = approxmatch_module.match_cost_grad(xyz1, xyz2, match)
    return [grad_1 * tf.expand_dims(tf.expand_dims(grad_cost, 1), 2),
            grad_2 * tf.expand_dims(tf.expand_dims(grad_cost, 1), 2), None]
