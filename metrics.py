import tensorflow as tf
import numpy as np


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    # print(tf.Session.run(mask))
    mask /= tf.reduce_mean(mask)
    # print(tf.Session.run(mask))
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def scoreReg(preds,labels):
    MSE = np.sum(np.power((labels - preds), 2)) / 202.
    R2 = 1 - MSE / np.var(preds)
    return R2
#testY是一维数组，predicY是二维数组，故需要将testY转换一下

def mse(true, pred):
    # return np.sum((true - pred)**2)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=true)
    loss = loss/202.
    return tf.reduce_mean(loss)

