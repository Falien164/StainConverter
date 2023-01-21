import tensorflow as tf


class Layers:
    @staticmethod
    def down_sample(filters, size, strides=2, apply_batch_norm=True):
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                                          kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
        if apply_batch_norm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())
        return result

    @staticmethod
    def up_sample(filters, size, strides=2, apply_dropout=False):
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding='same',
                                                   kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                                   use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
        return result

    @staticmethod
    def res_block(filters, size=3, strides=1):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding='same',
                                                   kernel_initializer=initializer, use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.ReLU())
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding='same',
                                                   kernel_initializer=initializer, use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        return result
