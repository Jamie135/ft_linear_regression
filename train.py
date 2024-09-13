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
        Ynorm = Y.copy()
        Ymin = np.min(Y)
        Ymax = np.max(Y)
        for i in range(len(Ynorm)):
            Ynorm[i] = (Y[i] - Ymin) / (Ymax - Ymin)
        W = np.hstack((Xnorm, np.ones(Xnorm.shape)))
        Thetas = np.random.randn(2, 1)
        m = len(X)
        # print(f"X:\n{X}\n\nY:\n{Y}\n\nXnorm:\n{Xnorm}\n\nYnorm:\n{Ynorm}\n\nW:\n{W}\n\nThetas:\n{Thetas}")
        return X, Y, Xnorm, Ynorm, W, Thetas, m
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(-1)


def gradient_descent():
    """train the model with linear regression"""

    X, Y, Xnorm, Ynorm, W, Thetas, m = init_parameters()
    learning_rate = 0.07
    iterations = 1000
    cost = np.array([0] * 1000, dtype=float)

    # gradient descent algorithm
    for i in range(iterations):
        estimate_price = np.dot(W, Thetas)
        error = estimate_price - Y

        # cost function J(t1, t0) = (1 / 2m) * sum((W.[[t1], [t0]] - Y)^2)
        # where t1 = Theta[1], t0 = Theta[0] and W = [[1, 1], ... , [0.17913321, 1]]
        cost[i] = (1 / (2 * m)) * np.sum(error ** 2)

        # the formula for simultaneous update of Thetas 
        # for each iterations is to substract tmp from Thetas
        # where tmp is calculated by multiplying the learning rate
        # to both d/dt1(J(t1, t0)) and d/dt0(J(t1, t0)) which represent
        # the partial derivative of the cost function with respect to t1 and t0
        tmp = learning_rate * (1 / m) * np.dot(W.T, error)
        Thetas -= tmp
    prediction = W.dot(Thetas)

    # print(f"X:\n{X}\n\nY:\n{Y}\n\nXnorm:\n{Xnorm}\n\nYnorm:\n{Ynorm}\n\nW:\n{W}\n\ncost:\n{cost}\n\nTheta:\n{Theta}\n\nprediction:\n{prediction}")
    return X, Y, Xnorm, Ynorm, iterations, cost, Thetas, prediction



def train():
    """train a dataset from data.csv to get the parameters"""

    X, Y, Xnorm, Ynorm,  iterations, cost, Thetas, prediction  = gradient_descent()
    try:
        parameters = {
            'X': X.tolist(),
            'Y': Y.tolist(),
            'Xnorm': Xnorm.tolist(),
            'Ynorm': Ynorm.tolist(),
            'iterations': iterations,
            'cost': cost.tolist(),
            'Theta0': Thetas[1][0],
            'Theta1': Thetas[0][0],
            'prediction': prediction.tolist()
        }
        with open("parameters.json", "w") as para:
            json.dump(parameters, para, indent=4)
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(-1)
    print("Parameters has been created, see file: parameters.json")


if __name__ == '__main__':
    train()
