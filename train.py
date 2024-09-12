import pandas as pd
import numpy as np


def train():
    data = pd.read_csv('data.csv')
    mileage = data[:, 0]
    price = data[:, 1]

    theta0 = 0
    theta1 = 0
    learning_rate = 0.01
    m = len(mileage)

    for _ in range(1000):
        estimate_price = theta0 + theta1 * mileage
        error = estimate_price - price
        tmp_theta0 = learning_rate * (1/m) * np.sum(error)
        tmp_theta1 = learning_rate * (1/m) * np.sum(error * mileage)
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

    with open('predictions.txt', 'w') as f:
        f.write(f'{theta0},{theta1}')


if __name__ == '__main__':
    train()
