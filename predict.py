import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_thetas():
    """get the theta values"""
    try:
        with open('thetas.json', 'r') as thetas_file:
            Thetas = json.load(thetas_file)
        theta0 = Thetas['Theta0']
        theta1 = Thetas['Theta1']
    except:
        theta0 = 0
        theta1 = 0
    return theta0, theta1


def get_mileage():
    while True:
        try:
            km = input("Mileage of your car: ")
            if not all(c in '-0123456789' for c in km):
                raise ValueError
            return float(km)
        except ValueError:
            print('Please enter a valid integer\n')
        except KeyboardInterrupt:
            print('\n\nExiting...')
            sys.exit(-1)


def predict():
    """predict the price of a car from its mileage"""

    theta0, theta1 = get_thetas()
    km_input = get_mileage()
    print(km_input)


if __name__ == '__main__':
    predict()