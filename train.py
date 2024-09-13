import os
import sys
import argparse
import pandas as pd
import numpy as np


# def check_file():
#     """check if data.csv exists or is correctly formatted"""

#     assert os.path.exists('data.csv'), "file doesn't exist."
#     data = pd.read_csv('data.csv')
#     x = np.array(data["km"])
#     y = np.array(data["price"])
#     print(x)
#     print(y)
#     return data


def dataset():
    """create a dataset from data.csv"""

    try:
        data = pd.read_csv('data.csv')
        X = np.array(data['km'], dtype=float)
        Y = np.array(data['price'], dtype=float)
        if np.isnan(X).any() or np.isnan(Y).any():
            raise ValueError("data.csv contains NaN values")
        Theta = np.random.randn(2, 1)
        # print(data)
        # print(X)
        # print(Y)
        # print(Theta)
        X = X.reshape(X.shape[0], 1)
        Y = Y.reshape(Y.shape[0], 1)
        Xnorm = X
        Xmin = np.min(X)
        Xmax = np.max(X)
        for i in range(len(Xnorm)):
            Xnorm[i] = (X[i] - Xmin) / (Xmax - Xmin)
        normX = np.hstack((Xnorm, np.ones(Xnorm.shape)))
        print(normX)
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(-1)


def train():
    """train a dataset from data.csv and save the model in predictions.txt"""

    dataset()
    # theta0 = 0
    # theta1 = 0
    # learning_rate = 0.07
    # m = len(mileage)

    # for _ in range(1000):
    #     estimate_price = theta0 + theta1 * mileage
    #     error = estimate_price - price
    #     tmp_theta0 = learning_rate * (1/m) * np.sum(error)
    #     tmp_theta1 = learning_rate * (1/m) * np.sum(error * mileage)
    #     theta0 -= tmp_theta0
    #     theta1 -= tmp_theta1

    # with open('predictions.txt', 'w') as f:
    #     f.write(f'{theta0},{theta1}')


if __name__ == '__main__':
    train()
