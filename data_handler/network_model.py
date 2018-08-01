'''
ecg_realtime_abnormal_detection
Created 1/08/18 by Matthew Lee
'''
import tensorflow as tf
import config


def instantiate_model():
    X = tf.placeholder(tf.float32, [None, config.network['feature_size'], config.network['feature_channels']])
    y = tf.placeholder(tf.float32, [None, config.network['labels']])
    print(X)
    # Plit the two channels for each convolution process
    dif, avg_dif = tf.unstack(X, axis=2)
    dif = tf.expand_dims(dif, axis=2)
    avg_dif = tf.expand_dims(avg_dif, axis=2)

    # Part level convolution

    dif_conv_1 = tf.layers.conv1d(dif, filters=8, kernel_size=6)
    dif_pool_1 = tf.layers.max_pooling1d(dif_conv_1, pool_size=2, strides=2)
    dif_conv_2 = tf.layers.conv1d(dif_pool_1, filters=16, kernel_size=6)
    dif_pool_2 = tf.layers.max_pooling1d(dif_conv_2, pool_size=2, strides=2)


    # Object level convolution

    avg_conv_1 = tf.layers.conv1d(avg_dif, filters=8, kernel_size=6)
    avg_pool_1 = tf.layers.max_pooling1d(avg_conv_1, pool_size=2, strides=2)
    avg_conv_2 = tf.layers.conv1d(avg_pool_1, filters=16, kernel_size=6)
    avg_pool_2 = tf.layers.max_pooling1d(avg_conv_2, pool_size=2, strides=2)

    # Concatenate

    dif_flat = tf.layers.Flatten()(dif_pool_2)
    avg_flat = tf.layers.Flatten()(avg_pool_2)
    conv_concat = tf.concat([dif_flat, avg_flat], axis=1)

    # MLP

    layer_1 = tf.layers.dense(conv_concat, 124)
    output_layer = tf.layers.dense(layer_1, config.network['labels'])
    output_softmax = tf.nn.softmax(output_layer)

    return X, y, output_layer, output_softmax
