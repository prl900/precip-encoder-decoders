import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import matplotlib.pyplot as plt

def create_dataset(samples, tile_size):
    margin = (tile_size - 1) / 2

    n = np.random.randint(0, high=40000, size=(samples))
    i = np.random.randint(margin, high=120 - margin, size=(samples))
    j = np.random.randint(margin, high=80 - margin, size=(samples))

    c = np.dstack((n, j, i))[0]


    x_tiles = np.zeros((samples, tile_size*tile_size*3), dtype=np.float32)
    y_tiles = np.zeros((samples), dtype=np.float32)
    for k, idx in enumerate(c):
        x_tiles[k, :] = x[idx[0], idx[1] - margin:idx[1] + (margin+1), idx[2] - margin:idx[2] + (margin+1), :].flatten()
        y_tiles[k] = y[idx[0], idx[1], idx[2]]

    return x_tiles, y_tiles


def predict(model, n, tile_size):
    margin = (tile_size - 1) / 2
    pred = np.zeros((80, 120), dtype=np.float32)

    for i in range(margin, 120-margin):
        for j in range(margin, 80-margin):
            pred[j, i] = model.predict([x[n, j-margin:j+(margin+1), i-margin:i+(margin+1), :].flatten()])[0]

    return pred


def mae(y, y_hat, tile_size):
    margin = (tile_size - 1) / 2

    return np.sum(np.absolute((y[margin:80-margin, margin:120-margin] - y_hat[margin:80-margin, margin:120-margin])))


def save_precip(name, y, y_hat):
    plt.imsave(name+"era", y, vmax=17, cmap='Blues')
    plt.imsave(name+"pred", y_hat, vmax=17, cmap='Blues')

if __name__ == "__main__":
    x = np.load("/Users/pablo/Downloads/3zlevels.npy")
    y = (np.load("/Users/pablo/Downloads/full_tp_1980_2016.npy") * 1000).clip(min=0)

    for tile_size in [3, 5, 7, 9, 11]:
        x_tiles, y_tiles = create_dataset(30000, tile_size)

        rf = RandomForestRegressor(max_depth=25, random_state=0)
        rf.fit(x_tiles, y_tiles)

        maes = 0
        for n in range(40000, 40100):
            y_hat = predict(rf, n, tile_size)
            save_precip("output/RF_{}".format(n), y[n, :], y_hat)
            maes += mae(y[n, :], y_hat, tile_size)

        print "Random Forest", tile_size, ":", maes/100.0

        ls = linear_model.Lasso(alpha=0.1)
        ls.fit(x_tiles, y_tiles)

        maes = 0
        for n in range(40000, 40100):
            y_hat = predict(ls, n, tile_size)
            save_precip("output/LASSO_{}".format(n), y[n, :], y_hat)
            maes += mae(y[n, :], y_hat, tile_size)

        print "LASSO", tile_size, ":", maes/100.0

        lr = linear_model.LinearRegression()
        lr.fit(x_tiles, y_tiles)

        maes = 0
        for n in range(40000, 40100):
            y_hat = predict(lr, n, tile_size)
            save_precip("output/LR_{}".format(n), y[n, :], y_hat)
            maes += mae(y[n, :], y_hat, tile_size)

        print "Linear Regression", tile_size, ":", maes/100.0
