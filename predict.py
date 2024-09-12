import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def predict():
    # parser = argparse.ArgumentParser(description='Predict the price of a car')
    # parser.add_argument('mileage', type=int, help='mileage of the car')
    # parser.add_argument('trained data', type=str, help='trained data for prediction', default=None)
    # args = parser.parse_args()

    try:
        with open('predictions.txt', 'r') as f:
            theta0, theta1 = map(float, f.read().split(','))
    except:
        theta0 = 0, theta1 = 0

    # theta = np.load('theta.npy')
    # mileage = args.mileage
    # price = theta[0] + theta[1] * mileage
    # print(f'Price of a car with {mileage} km: {price}')

    # plt.scatter(data['km'], data['price'], color='blue')
    # plt.plot(data['km'], theta[0] + theta[1] * data['km'], color='red')
    # plt.scatter(mileage, price, color='red')
    # plt.xlabel('Mileage')
    # plt.ylabel('Price')
    # plt.title('Linear regression')
    # plt.show()
    # Load the parameters

    # Prompt user for mileage
    mileage = float(input("Enter the mileage of the car: "))

    # Estimate price
    estimate_price = theta0 + theta1 * mileage
    print(f"The estimated price for a car with {mileage} mileage is: {estimate_price}")


if __name__ == '__main__':
    predict()
