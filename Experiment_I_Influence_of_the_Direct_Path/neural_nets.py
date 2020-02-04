import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Lambda, Concatenate, Conv2D
from keras.layers import Add, Multiply, Reshape, Dropout
from keras.layers import BatchNormalization, AveragePooling2D
from keras import backend as K
# from keras.utils import multi_gpu_model


# metric function
def dist(y_true, y_pred):
    return tf.reduce_mean((
        tf.sqrt(
            tf.square(tf.abs(y_pred[:, 0] - y_true[:, 0]))
            + tf.square(tf.abs(y_pred[:, 1] - y_true[:, 1]))
        )))


# build CNN based on DenseNet architecture
def build_cnn():
    nn_input = Input((64, 100, 2))

    num_complex_channels = 6
    num_antenna = 64
    num_sub = 100

    def k_mean(tensor):
        return K.mean(tensor, axis=2)

    # mean_input = Lambda(k_mean)(nn_input)
    # print(mean_input.get_shape())

    # complex to polar
    real = Lambda(lambda x: x[:, :, :, 0])(nn_input)
    imag = Lambda(lambda x: x[:, :, :, 1])(nn_input)

    real_squared = Multiply()([real, real])
    imag_squared = Multiply()([imag, imag])

    real_imag_squared_sum = Add()([real_squared, imag_squared])

    # amplitude
    def k_sqrt(tensor):
        r = K.sqrt(tensor)
        return r

    r = Lambda(k_sqrt)(real_imag_squared_sum)
    r = Reshape((num_antenna, num_sub, 1))(r)
    # print(r.get_shape())

    # phase
    def k_atan(tensor):
        import tensorflow as tf
        t = tf.math.atan2(tensor[0], tensor[1])
        return t

    t = Lambda(k_atan)([imag, real])
    t = Reshape((num_antenna, num_sub, 1))(t)
    # print(t.get_shape())

    polar_input = Concatenate()([r, t])

    # time domain
    def ifft(x):
        y = tf.complex(x[:, :, :, 0], x[:, :, :, 1])
        ifft = tf.spectral.ifft(y)
        return tf.stack([tf.math.real(ifft), tf.math.imag(ifft)], axis=3)

    time_input = Lambda(ifft)(nn_input)
    total_input = Concatenate()([nn_input, polar_input, time_input])

    # DenseNet CNN
    lay = Reshape((num_antenna, num_sub, num_complex_channels))(total_input)

    def dense_block(dense_in, num_channels, kernel_size):
        hidden_c = Conv2D(num_channels, kernel_size, padding='same',
                          activation='relu')(dense_in)
        hidden_v = Conv2D(num_channels, kernel_size, padding='same',
                          activation='relu')(hidden_c)
        for i in range(2):
            hidden_c = Concatenate()([hidden_c, hidden_v])
            hidden_v = Conv2D(num_channels, kernel_size, padding='same',
                              activation='relu')(hidden_c)
        # hidden012 = Concatenate()([hidden0, hidden1, hidden2])
        # hidden3 = Conv2D(num_channels, kernel_size, padding='same',
        #                  activation='relu',
        #                  kernel_regularizer=regularizers.l2(0.01))(hidden012)
        # hidden0123 = Concatenate()([hidden0, hidden1, hidden2, hidden3])
        # hidden4 = Conv3D(num_channels, kernel_size, padding='same')(hidden0123)
        hidden_v = BatchNormalization()(hidden_v)
        return hidden_v

    # size 64, 100, 6

    lay = dense_block(lay, 16, (1, 9))
    lay = AveragePooling2D((1, 5))(lay)

    lay = dense_block(lay, 16, (1, 9))
    lay = AveragePooling2D((1, 2))(lay)

    lay = dense_block(lay, 4, (1, 9))
    lay = dense_block(lay, 4, (1, 9))
    lay = AveragePooling2D((1, 5))(lay)

    enc_out = Flatten()(lay)

    encoder = Model(inputs=nn_input, outputs=enc_out)
    encoder.summary()

    return encoder


def build_fully_connected():
    nn_input = Input((512, ))
    dropout_rate = 0.0
    lay = Dense(256, activation="relu")(nn_input)
    lay = Dropout(dropout_rate)(lay)
    lay = Dense(64, activation="relu")(lay)
    lay = Dropout(dropout_rate)(lay)
    lay = Dense(32, activation="relu")(lay)
    # lay = Dropout(dropout_rate)(lay)

    model = Model(inputs=nn_input, outputs=lay)
    return model


def build_label():
    nn_input = Input((32, ))
    lay = Dense(2, activation="linear")(nn_input)

    model = Model(input=nn_input, outputs=lay)
    return model
