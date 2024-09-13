import sys
import json
import pandas as pd
import numpy as np


def init_parameters():
    """initialize the parameters for the model"""

    try:
        data = pd.read_csv('data.csv')
        X = np.array(data['km'], dtype=float)
        Y = np.array(data['price'], dtype=float)
        if np.isnan(X).any() or np.isnan(Y).any():
            raise ValueError("data.csv contains NaN values")
        X = X.reshape(X.shape[0], 1)
        Y = Y.reshape(Y.shape[0], 1)
        Xnorm = X.copy()
        Xmin = np.min(X)
        Xmax = np.max(X)
        for i in range(len(Xnorm)):
            Xnorm[i] = (X[i] - Xmin) / (Xmax - Xmin)
        Xnorm = np.hstack((Xnorm, np.ones(Xnorm.shape)))
        Thetas = np.random.randn(2, 1)
        m = len(X)
        # print(f"X:\n{X}\n\nY:\n{Y}\n\nXnorm:\n{Xnorm}\n\nTheta:\n{Theta}")
        return X, Y, Xnorm, Thetas, m
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(-1)


def linear_regression():
    """train the model with linear regression"""

    X, Y, Xnorm, Thetas, m = init_parameters()
    learning_rate = 0.07
    iterations = 1000
    cost = np.array([0] * 1000, dtype=float)

    # gradient descent
    for i in range(iterations):
        estimate_price = np.dot(Xnorm, Thetas)
        error = estimate_price - Y
        cost[i] = (1 / (2 * m)) * np.sum(error ** 2)
        tmp = learning_rate * (1 / m) * np.dot(Xnorm.T, error)
        Thetas -= tmp
    prediction = Xnorm.dot(Thetas)

    # print(f"X:\n{X}\n\nY:\n{Y}\n\ncost:\n{cost}\n\nTheta:\n{Theta}\n\nprediction:\n{prediction}")
    return X, Y, iterations, cost, Thetas, prediction



def train():
    """train a dataset from data.csv to get the model"""

    X, Y, iterations, cost, Thetas, prediction  = linear_regression()
    try:
        Thetas_dict = {
            'Theta0': Thetas[1][0],
            'Theta1': Thetas[0][0]
        }
        with open("thetas.json", "w") as Thetas_json:
            json.dump(Thetas_dict, Thetas_json, indent=4)
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(-1)
    print("Thetas parameter has been calculated, see file: thetas.json")


if __name__ == '__main__':
    train()
