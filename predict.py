import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


############################### Bonus ###############################
def plot_repartition(X, Y):
    """plot the data"""

    plt.scatter(X, Y, marker='+', label='Cars')
    plt.title('Prices of cars based on their mileage from data.csv')
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price (€)')
    plt.legend()
    plt.show()


def plot_prediction(X, Y, theta0, theta1, mileage, estimated_price):
    """plot the data and the prediction line"""

    # generate a range of mileage values from 0 to 250,000 km
    mileage_range = np.linspace(0, 250000, 500)
    normalized_mileage_range = (mileage_range - 22899) / (240000 - 22899)
    prediction_range = theta0 + theta1 * normalized_mileage_range

    plt.scatter(X, Y, marker='+', label='Cars')
    plt.plot(mileage_range, prediction_range, c='r', label='Prediction line')
    if mileage <= 250000:
        plt.scatter([mileage], [estimated_price], c='g', marker='o', label='Estimated price')
        plt.annotate(f'{estimated_price:.2f}€', (mileage, estimated_price), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price (€)')
    plt.title('Prediction of a car\'s price based on each mileage (0-250000km)')
    plt.legend()
    plt.show()


def plot_loss(iterations, cost):
    """plot the loss curve"""

    plt.plot(range(iterations), cost)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost History')
    plt.legend(['Loss curve'])
    plt.show()
#####################################################################


def get_parameters():
    """get all parameters from training"""

    try:
        data = pd.read_csv('data.csv')
        X = np.array(data['km'], dtype=float)
        Y = np.array(data['price'], dtype=float)
        if np.isnan(X).any() or np.isnan(Y).any():
            raise ValueError("data.csv contains NaN values")
        X = X.reshape(X.shape[0], 1)
        Y = Y.reshape(Y.shape[0], 1)
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(1)
    try:
        with open('parameters.json', 'r') as para:
            parameters = json.load(para)
        iterations = parameters['iterations']
        cost = np.array(parameters['cost'])
        theta1 = parameters['Theta1']
        theta0 = parameters['Theta0']
        prediction = np.array(parameters['prediction'])
    except:
        iterations = 0
        cost = 0
        theta0 = 0
        theta1 = 0
        prediction = 0
    # print(f"X:\n{X}\n\nY:\n{Y}\n\ncost:\n{cost}\n\nTheta:\n{theta0}, {theta1}\n\nprediction:\n{prediction}")
    return X, Y, iterations, cost, theta0, theta1, prediction


def get_mileage():
    """get the mileage input of the car"""

    while True:
        try:
            km = input("Mileage of your car: ")
            mileage = float(km)
            if mileage < 0:
                raise ValueError
            return mileage
        except ValueError:
            print('Please enter a valid positive integer\n')
        except KeyboardInterrupt:
            print('\n\nExiting...')
            sys.exit(-1)


def hypothesis(theta0, theta1):
    """calculate the estimated price of the car"""

    mileage = get_mileage()
    normalized = (mileage - 22899) / (240000 - 22899)
    # print(f"\nnormalized mileage: {normalized:.2f}\n\ntheta0: {theta0}\n\ntheta1: {theta1}\n")
    estimated_price = theta0 + theta1 * normalized
    return mileage, estimated_price


def predict():
    """predict the price of a car from its mileage"""

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--repartition", action="count", default=0, help="show the scatter plot of the data")
    parser.add_argument("-p", "--prediction", action="count", default=0, help="show the prediction line")
    parser.add_argument("-c", "--cost", action="count", default=0, help="show the cost history curve")
    args = parser.parse_args()

    X, Y, iterations, cost, theta0, theta1, prediction = get_parameters()

    if args.repartition >= 1:
        plot_repartition(X, Y)

    mileage, estimated_price = hypothesis(theta0, theta1)
    if estimated_price < 0:
        print("You should not sell your car, let her rest...")
        sys.exit(0)
    elif mileage > 250000:
        print("The car is too old for the prediction to be effective")
    print(f"Estimated price: {estimated_price:.2f}€")

    try:
        if isinstance(cost, np.ndarray) and isinstance(prediction, np.ndarray):
            if args.prediction >= 1:
                plot_prediction(X, Y, theta0, theta1, mileage, estimated_price)
            elif args.cost >= 1:
                plot_loss(iterations, cost)
    except KeyboardInterrupt:
        print('\n\nExiting...')
        sys.exit(-1)
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(1)


if __name__ == '__main__':
    predict()
