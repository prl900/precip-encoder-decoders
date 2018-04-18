from keras.models import Sequential
from keras import layers
from keras import models
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras import regularizers
import numpy as np
import pickle

from keras import backend as K
from keras.layers import Layer


class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(inputs, ksize=ksize, strides=strides, padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [dim // ratio[idx] if dim is not None else None for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with K.tf.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = K.tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (input_shape[0], input_shape[1] * self.size[0], input_shape[2] * self.size[1], input_shape[3])
            self.output_shape1 = output_shape

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
            batch_range = K.reshape(K.tf.range(output_shape[0], dtype='int32'), shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = K.tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = K.tf.size(updates)
            print "AAAAA", updates_size
            indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return mask_shape[0], mask_shape[1] * self.size[0], mask_shape[2] * self.size[1], mask_shape[3]



def get_segnet():
    concat_axis = 3
    pool_size = (2,2)

    inputs = layers.Input(shape = (80, 120, 3))

    conv_1 = BatchNormalization()(inputs)
    conv_1 = Conv2D(64, (3, 3), activation='relu', padding="same")(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_2 = Conv2D(64, (3, 3), activation='relu', padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Conv2D(128, (3, 3), activation='relu', padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_4 = Conv2D(128, (3, 3), activation='relu', padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Conv2D(256, (3, 3), activation='relu', padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_6 = Conv2D(256, (3, 3), activation='relu', padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_7 = Conv2D(256, (3, 3), activation='relu', padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Conv2D(512, (3, 3), activation='relu', padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_9 = Conv2D(512, (3, 3), activation='relu', padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_10 = Conv2D(512, (3, 3), activation='relu', padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)

    #pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)
    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size=(2, 3), strides=(2, 3))(conv_10)

    conv_11 = Conv2D(512, (3, 3), activation='relu', padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_12 = Conv2D(512, (3, 3), activation='relu', padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_13 = Conv2D(512, (3, 3), activation='relu', padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)

    #pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    #print("Build enceder done..")

    # decoder

    #unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])
    conv_14 = Conv2D(512, (3, 3), activation='relu', padding="same")(conv_13)
    conv_14 = BatchNormalization()(conv_14)
    conv_15 = Conv2D(512, (3, 3), activation='relu', padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_16 = Conv2D(512, (3, 3), activation='relu', padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)

    #unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])
    unpool_2 = MaxUnpooling2D((2,3))([conv_16, mask_4])

    conv_17 = Conv2D(512, (3, 3), activation='relu', padding="same")(unpool_2)
    #conv_17 = Conv2D(512, (3, 3), activation='relu', padding="same")(conv_10)
    conv_17 = BatchNormalization()(conv_17)
    conv_18 = Conv2D(512, (3, 3), activation='relu', padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_19 = Conv2D(256, (3, 3), activation='relu', padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = Conv2D(256, (3, 3), activation='relu', padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_21 = Conv2D(256, (3, 3), activation='relu', padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_22 = Conv2D(128, (3, 3), activation='relu', padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = Conv2D(128, (3, 3), activation='relu', padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_24 = Conv2D(64, (3, 3), activation='relu', padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = Conv2D(64, (3, 3), activation='relu', padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)

    conv_26 = Conv2D(1, (1, 1), activation='relu', padding="valid")(conv_25)
    #conv_26 = Reshape((80, 120, 1), input_shape=(80, 120, 1))(conv_26)

    #outputs = Activation(output_mode)(conv_26)
    outputs = conv_26
    print("Build decoder done..")

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(loss='mae', optimizer=Adam(lr=0.001), metrics=['mse'])
    print model.summary()

    return model


#x = np.load("/datasets/1980-2016/z_1980_2016.npy")
x = np.load("/datasets/10zlevels.npy")
y = 1000*np.expand_dims(np.load("/datasets/1980-2016/full_tp_1980_2016.npy"), axis=3)

idxs = np.arange(x.shape[0])
np.random.seed(0)
np.random.shuffle(idxs)

x = x[idxs, :, :, :]
y = y[idxs, :]

y_train = y[:40000, :]
y_test = y[40000:, :]

# Levels [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]

i = 0
j = 2
k = 6
x_train = x[:40000, :, :, [i,j,k]]
x_test = x[40000:, :, :, [i,j,k]]

model = get_segnet()
history = model.fit(x_train, y_train, epochs=50, verbose=1, validation_data=(x_test, y_test))
with open('trainHistoryDict_{}-{}-{}'.format(i, j, k), 'wb') as file_hist:
    pickle.dump(history.history, file_hist)

#model.save('/datasets/3chan_{}-{}-{}_.h5')

"""
    for i in range(10):
        rain_pred = model.predict(x_test[i:i+1,:])
        rain = y_test[i:i+1,:]

        print "Total pred", np.sum(rain_pred)
        print "Total", np.sum(rain)

        print "Mean pred", np.sum(rain_pred) / float(rain.shape[1] * rain.shape[2])
        print "Mean", np.sum(rain) / float(rain.shape[1] * rain.shape[2])

        mae = np.sum(rain_pred - rain)
        print "E", mae

        mae = np.sum(np.absolute(rain_pred - rain))
        print "AE", mae
        mae /= float(rain.shape[1] * rain.shape[2])
        print "MAE", mae
"""
