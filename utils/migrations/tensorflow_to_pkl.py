import pickle

import tensorflow as tf
from tensorflow.python.framework import ops

nn_distance_module = tf.load_op_library('tf_ops/libtf_nndistance.so')


def nn_distance(xyz1, xyz2):
    '''
    Computes the distance of nearest neighbors for a pair of point clouds
    input: xyz1: (batch_size,#points_1,3)  the first point cloud
    input: xyz2: (batch_size,#points_2,3)  the second point cloud
    output: dist1: (batch_size,#point_1)   distance from first to second
    output: idx1:  (batch_size,#point_1)   nearest neighbor from first to second
    output: dist2: (batch_size,#point_2)   distance from second to first
    output: idx2:  (batch_size,#point_2)   nearest neighbor from second to first
    '''
    return nn_distance_module.nn_distance(xyz1, xyz2)


@ops.RegisterGradient('NnDistance')
def _nn_distance_grad(op, grad_dist1, grad_idx1, grad_dist2, grad_idx2):
    xyz1 = op.inputs[0]
    xyz2 = op.inputs[1]
    idx1 = op.outputs[1]
    idx2 = op.outputs[3]
    return nn_distance_module.nn_distance_grad(xyz1, xyz2, grad_dist1, idx1, grad_dist2, idx2)


pickle_format = dict()

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('checkpoint/gcn.ckpt.meta')
    what = new_saver.restore(sess, 'checkpoint/gcn.ckpt')
    all_vars = tf.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    for v in all_vars:
        try:
            v_ = sess.run(v)
            pickle_format[v.name] = v_
        except:
            pass
    with open("result.pkl", "wb") as f:
        pickle.dump(pickle_format, f)
