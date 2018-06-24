from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pickle

model = Sequential()
model.add(Conv2D(64, (5, 5), activation='relu', padding='same', dilation_rate=4, input_shape=(256,256,1)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=4))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=4))
model.add(Conv2DTranspose(128, (3, 3), activation='relu', dilation_rate=4, padding='same'))
model.add(Conv2DTranspose(64, (3, 3), activation='relu', padding='same', dilation_rate=4))
model.add(Conv2DTranspose(1, (5, 5), activation='relu', padding='same', dilation_rate=4))

#model.add(Conv2D(64, (5, 5), strides=(2, 2), activation='relu', padding='same', dilation_rate=4, input_shape=(256,256,1)))
#model.add(Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same', dilation_rate=4))
#model.add(Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', dilation_rate=4))
#model.add(Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', dilation_rate=4, padding='same'))
#model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same', dilation_rate=4))
#model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), activation='relu', padding='same', dilation_rate=4))

#model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['mse'])
model.compile(loss='mean_absolute_error', optimizer=Adam(lr=0.001), metrics=['mae'])

print model.summary()

x = np.load("/datasets/trans_dataset/x.npy")
y = np.load("/datasets/trans_dataset/y.npy")

print(x.shape)
print(y.shape)

plt.imsave('test.png', y[4500,:,:,0], vmax=1.0, cmap='Blues')
print "real", y[4500,:,:,0].max()
print "real", y[4501,:,:,0].max()
print "real", y[4502,:,:,0].max()
print "real", y[4503,:,:,0].max()
print "real", y[4504,:,:,0].max()
print "real", y[4505,:,:,0].max()
print "real", y[4506,:,:,0].max()
print "real", y[4507,:,:,0].max()
print "real", y[4508,:,:,0].max()
print "real", y[4509,:,:,0].max()
x_train = x[:4000, :]
y_train = y[:4000, :]

x_test = x[4000:, :]
y_test = y[4000:, :]

print x_train.shape
print y_train.shape

print x_test.shape
print y_test.shape

history = model.fit(x_train, y_train, epochs=15, verbose=1, validation_data=(x_test, y_test))
""" 
with open('trainHistoryDict_{}'.format(i), 'wb') as file_pi:
pickle.dump(history.history, file_pi)
"""

#model.save('rain_predictor.h5')
#rain = model.predict(x[45000:45001,:])/1000
rain = model.predict(x[4500:4501,:])
plt.imsave('test_pred.png', rain[0,:,:,0], vmax=1.0, cmap='Blues')
print "pred", rain[0,:,:,0].max()
rain = model.predict(x[4501:4502,:])
print "pred", rain[0,:,:,0].max()
rain = model.predict(x[4502:4503,:])
print "pred", rain[0,:,:,0].max()
rain = model.predict(x[4503:4504,:])
print "pred", rain[0,:,:,0].max()
rain = model.predict(x[4504:4505,:])
print "pred", rain[0,:,:,0].max()
rain = model.predict(x[4505:4506,:])
print "pred", rain[0,:,:,0].max()
rain = model.predict(x[4506:4507,:])
print "pred", rain[0,:,:,0].max()
rain = model.predict(x[4507:4508,:])
print "pred", rain[0,:,:,0].max()
rain = model.predict(x[4508:4509,:])
print "pred", rain[0,:,:,0].max()
rain = model.predict(x[4509:4510,:])
print "pred", rain[0,:,:,0].max()
rain = model.predict(x[4505:4506,:])
print "pred", rain[0,:,:,0].max()
#print rain[:,10]
