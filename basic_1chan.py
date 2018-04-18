from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pickle

def GetModel():
    model = Sequential()
    model.add(BatchNormalization(axis=3, input_shape=(80, 120, 1)))
    model.add(Conv2D(64, (5, 5), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), activation='relu', padding='same'))

    #model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['mse'])
    model.compile(loss='mean_absolute_error', optimizer=Adam(lr=0.001), metrics=['mae'])

    print model.summary()

    return model

x = np.load("/datasets/10zlevels.npy")
y = 1000*np.expand_dims(np.load("/datasets/1980-2016/full_tp_1980_2016.npy"), axis=3)

print(x.shape)
print(y.shape)

plt.imsave('rain.png', y[45000,:,:,0], cmap='Blues')

for i in range(10):
    x_train = x[:40000, :, :, i:i+1]
    y_train = y[:40000, :]

    x_test = x[40000:, :, :, i:i+1]
    y_test = y[40000:, :]

    print x_train.shape
    print y_train.shape

    print x_test.shape
    print y_test.shape

    model = GetModel()
    history = model.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_test, y_test))
    """
    with open('trainHistoryDict_{}'.format(i), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    """

    #model.save('rain_predictor.h5')
    #rain = model.predict(x[45000:45001,:])/1000
    rain = model.predict(x[45000:45001,:, :, i:i+1])/1000
    plt.imsave('rain_pred{}.png'.format(i), rain[0,:,:,0], cmap='Blues')
    #print rain[:,10]
