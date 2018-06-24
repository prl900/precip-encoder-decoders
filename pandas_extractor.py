from keras.models import load_model
#from keras import layers
#from keras.layers import Layer
#from keras import models
import matplotlib.pyplot as plt
import numpy as np
import sys

lat_bounds = (75, 15.75)
lon_bounds = (-50, 39.25)

cities = ["Rome", "Dublin", "Moscow", "Paris", "Marrakech", "Cairo", "Ponta Delgada", "Stockholm", "Reykjavik", "Bucharest", "Tamanrasset"]
coords = [(41.9028, 12.4964), (53.3498, -6.2603), (55.7558, 37.6173), (48.8566, 2.3522), 
          (31.6295, -7.9811), (30.0444, 31.2357), (37.7428, -25.6806), (59.3293, 18.0686), (64.1265, -21.8174), (44.4268, 26.1025), (22.788889, 5.525556) ]

def get_geoindex(lat, lon):
    lats = np.linspace(lat_bounds[0], lat_bounds[1], num=80)
    lons = np.linspace(lon_bounds[0], lon_bounds[1], num=120)
    lat_idx = (np.abs(lats-lat)).argmin()
    lon_idx = (np.abs(lons-lon)).argmin()

    return lat_idx, lon_idx

for n in range(11):
    j, i = get_geoindex(*coords[n])
    print cities[n], j, i

#sys.exit(0)

x = np.load("/datasets/10zlevels.npy")
y = 1000*np.expand_dims(np.load("/datasets/1980-2016/full_tp_1980_2016.npy"), axis=3)
print "data loaded"
print x.shape

levels = [0,2,6]

#model = load_model('/datasets/simple_model_0_2_6.h5')
model = load_model('/datasets/vgg16_0_2_6.h5')
#model = load_model('/datasets/unet1_0-2-6_.h5')

rain_table = np.zeros((6000, 22))

for n in range(0, 6000):
    if n%100 == 0:
        print n

    geop = x[n][:,:,levels]
    geop = geop[np.newaxis, ...]
    out = model.predict(geop)#/1000

    """ 
    print(n)
    print "CNN", np.sum(out[0, :, :, 0]), np.mean(out[0, :, :, 0]), np.max(out[0, :, :, 0])
    print "ERAI", np.sum(y[n, :, :, 0]), np.mean(y[n, :, :, 0]), np.max(y[n, :, :, 0])

    if n < 5050: 
        plt.imsave('out_era_{}.png'.format(n), y[n,:,:,0], cmap='Blues') 
        plt.imsave('out_{}.png'.format(n), out[0,:,:,0], cmap='Blues') 
    """
    for idx, city in enumerate(cities):
        j, i = get_geoindex(*coords[idx])
        rain_table[n, 2*idx] = out[0, j, i, 0]
        rain_table[n, 1+2*idx] = y[n, j, i, 0]

np.save("cities_vgg16", rain_table)
print rain_table

"""
    plt.imsave('in_{}.png'.format(i), x[i,:,:,0], cmap='jet') 
    plt.imsave('out_{}.png'.format(i), y[i,:,:,0], cmap='Blues') 
    plt.imsave('vgg16_{}.png'.format(i), rain[0,:,:,0], cmap='Blues') 
    rain = segnet.predict(geop)/1000
    plt.imsave('segnet_{}.png'.format(i), rain[0,:,:,0], cmap='Blues') 
    rain = unet.predict(geop)/1000
    plt.imsave('unet_{}.png'.format(i), rain[0,:,:,0], cmap='Blues') 
"""
