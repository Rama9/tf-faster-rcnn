from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

def weighed_logistic_loss(scores, labels, pos_loss_mult=1.0, neg_loss_mult=1.0):
    # Apply different weights to loss of positive samples and negative samples
    # positive samples have label 1 while negative samples have label 0
    loss_mult = tf.add(tf.mul(labels, pos_loss_mult-neg_loss_mult), neg_loss_mult)

    # Classification loss as the average of weighed per-score loss
    cls_loss = tf.reduce_mean(tf.mul(
        tf.nn.sigmoid_cross_entropy_with_logits(scores, labels),
        loss_mult))

    return cls_loss

def l2_regularization_loss(variables, weight_decay):
    l2_losses = [tf.nn.l2_loss(var) for var in variables]
    total_l2_loss = weight_decay * tf.add_n(l2_losses)
    return total_l2_loss

def smooth_l1_regression_loss(scores, labels, thres=1.0):
    # L1(x) = 0.5x^2 (|x|<thres)
    # L1(x) = |x|-0.5 (|x|>=thres)
    diff =  tf.abs(scores - labels)

    pos_num = tf.gather_nd(diff, tf.cast(tf.where(diff<thres), tf.int32))
    neg_num = tf.gather_nd(diff, tf.cast(tf.where(diff>=thres), tf.int32))

    loss = tf.reduce_sum(0.5*pos_num*pos_num) + tf.reduce_sum(neg_num-0.5)

    num_sample = scores.get_shape().as_list()[0]
    if num_sample is not None:
        loss /= float(num_sample)
    return loss

def ranking_loss(scores, margin=1.0):
    # first column as positive, other columns are negative in scores
    B, num_column = scores.get_shape().as_list()
    num_neg = num_column-1
    loss_vec = tf.zeros(B, tf.float32)
    for i in range(num_neg):
        loss_vec += tf.maximum(0.0, margin-scores[:, 0]+scores[:, 1+i])
    return loss_vec