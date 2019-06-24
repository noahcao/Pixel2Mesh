import tensorflow as tf

from tf_ops.tf_nndistance import nn_distance


def laplace_coord(pred, lape_idx):
    vertex = tf.concat([pred, tf.zeros([1, 3])], 0)
    indices = lape_idx[:, :8]
    weights = tf.cast(lape_idx[:, -1], tf.float32)

    weights = tf.tile(tf.reshape(tf.reciprocal(weights), [-1, 1]), [1, 3])
    laplace = tf.reduce_sum(tf.gather(vertex, indices), 1)
    laplace = tf.subtract(pred, tf.multiply(laplace, weights))
    return laplace


def laplace_loss(pred1, pred2, lape_idx):
    # laplace term
    lap1 = laplace_coord(pred1, lape_idx)
    lap2 = laplace_coord(pred2, lape_idx)
    return tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(lap1, lap2)), 1)) * 1500


def move_loss(pred1, pred2):
    return tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(pred1, pred2)), 1)) * 100


def unit(tensor):
    return tf.nn.l2_normalize(tensor, dim=1)


def mesh_loss(pred, labels, edges):
    gt_pt = labels[:, :3]  # gt points
    gt_nm = labels[:, 3:]  # gt normals

    # edge in graph
    nod1 = tf.gather(pred, edges[:, 0])
    nod2 = tf.gather(pred, edges[:, 1])
    edge = tf.subtract(nod1, nod2)

    # edge length loss
    edge_length = tf.reduce_sum(tf.square(edge), 1)
    edge_loss = tf.reduce_mean(edge_length) * 300

    # chamer distance
    dist1, idx1, dist2, idx2 = nn_distance(gt_pt, pred)
    point_loss = (tf.reduce_mean(dist1) + 0.55 * tf.reduce_mean(dist2)) * 3000

    # normal cosine loss
    normal = tf.gather(gt_nm, tf.squeeze(idx2, 0))
    normal = tf.gather(normal, edges[:, 0])
    cosine = tf.abs(tf.reduce_sum(tf.multiply(unit(normal), unit(edge)), 1))
    # cosine = tf.where(tf.greater(cosine,0.866), tf.zeros_like(cosine), cosine) # truncated
    normal_loss = tf.reduce_mean(cosine) * 0.5

    total_loss = point_loss + edge_loss + normal_loss
    return total_loss


def gcn_loss(target, pred):
    loss = 0
    for i in range(3):
        loss += mesh_loss(pred["outputs"][i], target["labels"], target["edges_%d" % i])
    loss += .1 * laplace_loss(target["features"], pred["outputs"][0], target["lape_idx_0"])
    for i in range(1, 3):
        loss += laplace_loss(pred["outputs_unpool_%d" % (i - 1)],
                             pred["outputs"][i], target["lape_idx_%d" % i]) + \
                move_loss(pred["outputs_unpool"][i - 1], pred["outputs"][i])
    return loss
