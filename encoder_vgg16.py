from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPooling2D, Dropout
from keras.optimizers import Adam, SGD
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
import itertools
import pickle

"""
def step_decay(epoch):
    if epoch < 25:
         return 0.01
    if epoch < 50:
         return 0.005
    else:
         return 0.001
"""
 
def get_vgg16():
    model = Sequential()

    # Encoder
    # Block 1
    model.add(BatchNormalization(axis=3, input_shape=(80, 120, 3)))
    #model.add(Dropout(.2))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block1_conv1', input_shape=(80,120,3)))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block2_conv1'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block3_conv1'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block3_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block4_conv1'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block4_conv2'))
    model.add(MaxPooling2D((2, 3), strides=(2, 3), name='block4_pool'))


    # Block 5
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block5_conv1'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block5_conv2'))

    # Decoder
    # Block 6
    model.add(UpSampling2D((2, 3), name='block6_upsampl'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block6_conv1'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block6_conv2'))

    # Block 7
    model.add(UpSampling2D((2, 2), name='block7_upsampl'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block7_conv1'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block7_conv2'))

    # Block 8
    model.add(UpSampling2D((2, 2), name='block8_upsampl'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block8_conv1'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block8_conv2'))

    # Block 9
    model.add(UpSampling2D((2, 2), name='block9_upsampl'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block9_conv1'))
    model.add(BatchNormalization(axis=3))
    #model.add(Dropout(.2))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block9_conv2'))

    # Output
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(1, (1, 1), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block10_conv1'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mae', optimizer=sgd, metrics=['mse'])
    #model.compile(loss='mae', optimizer=Adam(lr=0.001), metrics=['mse'])
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

model = get_vgg16()
history = model.fit(x_train, y_train, epochs=50, verbose=1, validation_data=(x_test, y_test))
with open('trainHistoryDict_vgg16_{}-{}-{}'.format(i, j, k), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
   
model.save('/datasets/vgg16_0_2_6.h5')

