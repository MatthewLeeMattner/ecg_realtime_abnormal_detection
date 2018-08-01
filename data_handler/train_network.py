'''
ecg_realtime_abnormal_detection
Created 1/08/18 by Matthew Lee
'''
import tensorflow as tf
import math

import config
from utils import log, v_log
from format_data import get_data, get_train_test
from network_model import instantiate_model as model

X_placeholder, y_placeholder, output, output_soft = model()


def calculate_accuracy(y, y_):
    correct = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    return tf.reduce_mean(tf.cast(correct, 'float'))


def calculate_loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))


def calculate_backpropagation(loss, lr):
    return tf.train.AdamOptimizer(lr).minimize(loss)


def train(X, y, epochs=config.train['epochs'], lr=config.train['learning_rate'], batch=config.train['batch']):
    loss = calculate_loss(output, y)
    backprop = calculate_backpropagation(loss, lr)
    accuracy = calculate_accuracy(y, output)

    X_train, X_test, y_train, y_test = get_train_test(*get_data())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            epoch_loss = 0
            for i in range(int(math.ceil(len(X_train)/batch))):
                if len(X_train) < i * batch + batch:
                    X_batch = X_train[i*batch:i*batch+batch]
                    y_batch = y_train[i*batch:i*batch+batch]
                else:
                    X_batch = X_train[i*batch:]
                    y_batch = y_train[i*batch:]
                l, _ = sess.run([loss, backprop], feed_dict={
                    X: X_batch,
                    y: y_batch
                })
                v_log("Batch loss: {}".format(l))
                epoch_loss += l
            log("Epoch {} loss: {}".format(e, epoch_loss))

            acc = sess.run(accuracy, feed_dict={
                X: X_test,
                y: y_test
            })
            print("Epoch {} accuracy: {}".format(e, acc))




if __name__ == "__main__":
    train(X_placeholder, y_placeholder)