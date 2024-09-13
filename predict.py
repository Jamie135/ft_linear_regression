import os
import sys
import json
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt


############################### Bonus ###############################
# def plot_prediction(X, Y, prediction, mileage, estimated_price):
#     """plot the data and the prediction line"""

#     plt.scatter(X, Y, marker='+')
#     plt.plot(X, prediction, c='r')
#     plt.scatter([mileage], [estimated_price], c='g', marker='o', label='Estimated price')
#     plt.annotate(f'{estimated_price:.2f}€', (mileage, estimated_price), textcoords="offset points", xytext=(0,10), ha='center')
#     plt.xlabel('Mileage (km)')
#     plt.ylabel('Price (€)')
#     plt.title('Prediction of a car\'s price based on mileage')
#     plt.legend(['Cars', 'Prediction', 'Estimated price'])
#     plt.show()

def plot_prediction(X, Y, theta0, theta1, mileage, estimated_price):
    """plot the data and the prediction line"""

    # generate a range of mileage values from 0 to 250,000 km
    mileage_range = np.linspace(0, 250000, 500)
    normalized_mileage_range = (mileage_range - 22899) / (240000 - 22899)
    prediction_range = theta0 + theta1 * normalized_mileage_range

    plt.scatter(X, Y, marker='+', label='Actual data')
    plt.plot(mileage_range, prediction_range, c='r', label='Prediction line')
    if mileage <= 250000:
        plt.scatter([mileage], [estimated_price], c='g', marker='o', label='Estimated price')
        plt.annotate(f'{estimated_price:.2f}€', (mileage, estimated_price), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price (€)')
    plt.title('Prediction of a car\'s price based on mileage')
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


def precision(Y, prediction):
	"""calculate the coefficient of determination"""

	a = ((Y - prediction) ** 2).sum()
	b = ((Y - Y.mean()) ** 2).sum()
	coef = (1 - a / b) * 100
	print("The precision is equal to: {:.{prec}f}%".format(coef, prec=2))
	sys.exit(0)
#####################################################################


def get_parameters():
    """get the theta values"""

    try:
        with open('parameters.json', 'r') as para:
            parameters = json.load(para)
        X = np.array(parameters['X'])
        Y = np.array(parameters['Y'])
        Xnorm = np.array(parameters['Xnorm'])
        Ynorm = np.array(parameters['Ynorm'])
        iterations = parameters['iterations']
        cost = np.array(parameters['cost'])
        theta0 = parameters['Theta0']
        theta1 = parameters['Theta1']
        prediction = np.array(parameters['prediction'])
    except:
        X = 0
        Y = 0
        Xnorm = 0
        Ynorm = 0
        iterations = 0
        cost = 0
        theta0 = 0
        theta1 = 0
        prediction = 0
    # print(f"X:\n{X}\n\nY:\n{Y}\n\nXnorm:\n{Xnorm}\n\nYnorm:\n{Ynorm}\n\nW:\n{W}\n\ncost:\n{cost}\n\nTheta:\n{theta0}, {theta1}\n\nprediction:\n{prediction}")
    return X, Y, Xnorm, Ynorm, iterations, cost, theta0, theta1, prediction


def get_mileage():
    """get the mileage input of the car"""

    while True:
        try:
            km = input("Mileage of your car: ")
            mileage = float(km)
            if mileage > 409811:
                print("You should not sell your car, let her rest...")
                sys.exit(0)
            if mileage < 0:
                raise ValueError
            return mileage
        except ValueError:
            print('Please enter a valid positive integer\n')
        except KeyboardInterrupt:
            print('\n\nExiting...')
            sys.exit(-1)


def linear_regression(theta0, theta1):
    """calculate the estimated price of the car"""

    mileage = get_mileage()
    normalized = (mileage - 22899) / (240000 - 22899)
    # print(f"\nnormalized mileage: {normalized:.2f}\n\ntheta0: {theta0}\n\ntheta1: {theta1}\n")
    estimated_price = theta0 + theta1 * normalized
    return mileage, estimated_price


def predict():
    """predict the price of a car from its mileage"""

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prediction", action="count", default=0, help="show the prediction line")
    parser.add_argument("-c", "--cost", action="count", default=0, help="show the cost history curve")
    parser.add_argument("-d", "--determination", action="count", default=0, help="show the coefficient determination")
    args = parser.parse_args()

    X, Y, Xnorm, Ynorm, iterations, cost, theta0, theta1, prediction = get_parameters()
    plt.scatter(X, Y, marker='+')
    plt.title('Prices of cars based on mileage')
    plt.legend(['Cars', 'Prediction', 'Estimated price'])
    plt.show()
    mileage, estimated_price = linear_regression(theta0, theta1)
    print(f"Estimated price: {estimated_price:.2f}€")

    try:
        if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray) and isinstance(cost, np.ndarray) and isinstance(prediction, np.ndarray):
            if args.prediction >= 1:
                plot_prediction(X, Y, theta0, theta1, mileage, estimated_price)
            elif args.cost >= 1:
                plot_loss(iterations, cost)
            elif args.determination >= 1:
                precision(Y, prediction)
            
    except KeyboardInterrupt:
        print('\n\nExiting...')
        sys.exit(-1)
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(-1)
    finally:
        # delete the __pycache__ directory
        pycache_dir = os.path.join(os.path.dirname(__file__), '__pycache__')
        if os.path.exists(pycache_dir):
            shutil.rmtree(pycache_dir)


if __name__ == '__main__':
    predict()
