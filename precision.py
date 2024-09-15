import sys
import json
import numpy as np
import pandas as pd


def get_parameters():
    """get Y and prediction parameters from training"""

    try:
        data = pd.read_csv('data.csv')
        Y = np.array(data['price'], dtype=float)
        if np.isnan(Y).any():
            raise ValueError("data.csv contains NaN values")
        Y = Y.reshape(Y.shape[0], 1)
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(-1)
    try:
        with open('parameters.json', 'r') as para:
            parameters = json.load(para)
        prediction = np.array(parameters['prediction'])
    except:
        print("Your data is not trained")
        exit(0)
    return Y, prediction


def precision():
    """calculate the coefficient of determination and display it in percentage"""

    # R2 = 1 - (SSres /  SStot)
    Y, prediction = get_parameters()
    ssr = ((Y - prediction) ** 2).sum()
    sst = ((Y - Y.mean()) ** 2).sum()
    r2 = (1 - ssr / sst)
    # print(f"Ymean: {Y.mean()}")
    print("The precision is equal to: {:.{prec}f}%".format(r2 * 100, prec=2))


if __name__ == '__main__':
    precision()
