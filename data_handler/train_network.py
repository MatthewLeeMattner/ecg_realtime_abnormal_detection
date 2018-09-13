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
    with tf.name_scope("accuracy"):
        correct = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        accuracy_summary = tf.summary.scalar("Accuracy", accuracy)
    return accuracy, accuracy_summary


def calculate_loss(logits, labels):
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        loss_summary = tf.summary.scalar("Loss", loss)
    return loss, loss_summary


def calculate_backpropagation(loss, lr, global_step_tensor):
    return tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step_tensor)


def train(X, y, epochs=config.train['epochs'], lr=config.train['learning_rate'], batch=config.train['batch']):
    global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    loss, loss_summary = calculate_loss(output, y)
    backprop = calculate_backpropagation(loss, lr, global_step_tensor)
    accuracy, accuracy_summary = calculate_accuracy(y, output)

    merged = tf.summary.merge_all()


    X_train, X_test, y_train, y_test = get_train_test(*get_data())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


        train_writer = tf.summary.FileWriter(config.train['tensorboard'] + "/" + config.train['name'] + "/train", sess.graph)
        test_writer = tf.summary.FileWriter(config.train['tensorboard'] + "/" + config.train['name'] + "/test", sess.graph)

        for e in range(epochs):
            epoch_loss = 0
            for i in range(int(math.ceil(len(X_train)/batch))):
                if len(X_train) < i * batch + batch:
                    X_batch = X_train[i*batch:i*batch+batch]
                    y_batch = y_train[i*batch:i*batch+batch]
                else:
                    X_batch = X_train[i*batch:]
                    y_batch = y_train[i*batch:]
                l, _, summary = sess.run([loss, backprop, loss_summary], feed_dict={
                    X: X_batch,
                    y: y_batch
                })
                train_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step_tensor))
                v_log("Batch loss: {}".format(l))
                epoch_loss += l
            log("Epoch {} loss: {}".format(e, epoch_loss))

            acc, summary = sess.run([accuracy, accuracy_summary], feed_dict={
                X: X_test,
                y: y_test
            })
            test_writer.add_summary(summary, global_step=e)


            print("Epoch {} accuracy: {}".format(e, acc))




if __name__ == "__main__":
    train(X_placeholder, y_placeholder)