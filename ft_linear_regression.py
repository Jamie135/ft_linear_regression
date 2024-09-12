import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import shutil


class MyLinearRegression():

    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.0001, max_iter=500_000):
        if not isinstance(alpha, float) or alpha < 0.0:
            return None
        elif not isinstance(max_iter, int) or max_iter < 0:
            return None
        self.thetas = np.array(thetas)
        self.alpha = alpha
        self.max_iter = max_iter


    def predict_(self, x):
        """
        Computes the prediction vector y_hat from two non-empty numpy.array.
        
        Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.
        
        Return:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not matching.
        None if x or theta is not of expected type.
        
        Raises:
        This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray):
            return None
        elif x.size == 0:
            return None
        m = x.shape[0]
        n = x.shape[1]
        if self.thetas.shape != (n + 1, 1):
            return None
        # Add a column of ones to x to account for theta0
        X_prime = np.c_[np.ones(m), x]
        return X_prime @ self.thetas


    def gradient_(self, x, y):
        """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
        The three arrays must have the compatible dimensions.
        Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector (n +1) * 1.
        Return:
        The gradient as a numpy.array, a vector of dimensions n * 1,
        containing the result of the formula for all j.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible dimensions.
        None if x, y or theta is not of expected type.
        Raises:
        This function should not raise any Exception.
        """
        for array in [x, y]:
            if not isinstance(array, np.ndarray):
                return None
        m, n = x.shape
        if m == 0 or n == 0:
            return None
        elif y.shape != (m, 1):
            return None
        elif self.thetas.shape != (n + 1, 1):
            return None
        X_prime = np.c_[np.ones(m), x]
        return (X_prime.T @ (X_prime @ self.thetas - y)) / m


    def fit_(self, x, y):
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
        x: has to be a numpy.array, a matrix of dimension m * n:
        (number of training examples, number of features).
        y: has to be a numpy.array, a vector of dimension m * 1:
        (number of training examples, 1).
        theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
        (number of features + 1, 1).
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
        Return:
        new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
        None if there is a matching dimension problem.
        None if x, y, theta, alpha or max_iter is not of expected type.
        Raises:
        This function should not raise any Exception.
        """
        for arr in [x, y]:
            if not isinstance(arr, np.ndarray):
                return None
        m, n = x.shape
        if m == 0 or n == 0:
            return None
        if y.shape != (m, 1):
            return None
        elif self.thetas.shape != ((n + 1), 1):
            return None
        for _ in self.ft_progress(range(self.max_iter)):
            gradient = self.gradient_(x, y)
            if gradient is None:
                return None
            if all(__ == 0. for __ in gradient):
                break
            self.thetas -= self.alpha * gradient
        return self.thetas


    def mse_elem(self, y, y_hat) -> np.ndarray:
        return (y_hat - y) ** 2


    def mse_(self, y, y_hat) -> float:
        if any(not isinstance(_, np.ndarray) for _ in [y, y_hat]):
            return None
        m = y.shape[0]
        if m == 0 or y.shape != (m, 1) or y_hat.shape != (m, 1):
            return None
        J_elem = self.mse_elem(y, y_hat)
        return J_elem.mean()


    def minmax(x):
        if not isinstance(x, np.ndarray):
            return None
        if x.size == 0:
            return None
        if x.ndim != 1:
            x = x.reshape(-1)
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    

    def loss_elem_(self, y, y_hat):
        """
        Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
        y: has to be a numpy.array, a two-dimensional array of shape m * 1.
        y_hat: has to be a numpy.array, a two-dimensional array of shape m * 1.
        Returns:
        J_elem: numpy.array, an array of dimension (number of the training examples, 1).
        None if there is a dimension matching problem.
        None if any argument is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        if y.shape != y_hat.shape or y.size == 0 or y_hat.size == 0:
            return None
        if y.ndim != y_hat.ndim or y.size != y_hat.size:
            return None

        J_elem = (y_hat - y) ** 2
        return J_elem
    

    def loss_(self, y, y_hat):
        """Computes the half mean squared error of two non-empty numpy.array, without any for loop.
        The two arrays must have the same dimensions.
        Args:
        y: has to be an numpy.array, a one-dimensional array of size m.
        y_hat: has to be an numpy.array, a one-dimensional array of size m.
        Returns:
        The half mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
        Raises:
        This function should not raise any Exceptions.
        """
        J = self.loss_elem_(y, y_hat)
        if J is None:
            return None
        return np.mean(J) / 2
    

    # def ft_progress(self, iterable):
    #     total = len(iterable)
    #     start_time = time.time()
    #     # enumerate parcour iterable tout en gardant une trace de l'index de chaque élément
    #     for i, val in enumerate(iterable, start=1):
    #         # val represente l'element actuel de iterable
    #         # yield val sert à retourner l'élément de iterable sans terminer la fonction
    #         # elle renvoie un résultat au générateur appelant et conserve l'état de la fonction 
    #         # pour que l'exécution puisse reprendre là où elle s'est arrêtée lors du prochain appel 
    #         yield val

    #         # proportion de la barre parcourue
    #         progress = i / total

    #         # le temps écoulé depuis le début de l'itération
    #         elapsed_time = time.time() - start_time

    #         # calcul pour l'estimation du temps restant
    #         if progress > 0:
    #             eta = elapsed_time / progress * (1 - progress)
    #         else:
    #             eta = 0

    #         # '=' * int(progress * 20) calcule le nombre de symboles '='
    #         # ' ' * (19 - int(progress * 20)) calcule le nombre d'espaces nécessaires pour compléter la barre
    #         progress_bar = '[' + '=' * int(progress * 20) + '>' + ' ' * (19 - int(progress * 20)) + ']'

    #         # affiche la progression de la barre formatté comme dans le sujet
    #         #'\r' permet de overwrite la ligne recente de la progression
    #         print(f"ETA: {eta:.2f}s [{int(progress * 100)}%]{progress_bar} {i}/{total} | elapsed time {elapsed_time:.2f}s", end='\r')
    #     print()
    

    def ft_progress(self, lst):
        """
        Simulate a progress bar for iterating through a range.

        Args:
            lst (range): The range to iterate through.

        Yields:
            Any: The current item from the range.
            is a keyword in Python used in the context of creating generators.
            Generators are a way to create iterators, which are objects used to
            iterate over a sequence of values without having to store all those
            values in memory at once. Instead of generating allvalues and returning
            them in one go, a generator yields one value at a time whenever the
            yield statement is encountered.
        """
        total = len(lst)
        terminal_width = shutil.get_terminal_size().columns - 30
        progress_bar_width = terminal_width - 10
        # enumerate parcour lst tout en gardant
        # une trace de l'index de chaque élément
        for i, val in enumerate(lst, start=1):

            # val represente l'element actuel de lst
            # yield val sert à retourner l'élément de lst sans terminer la fonction
            # elle renvoie un résultat au générateur appelant
            # et conserve l'état de la fonction pour que l'exécution
            # puisse reprendre là où elle s'est arrêtée lors du prochain appel
            yield val

            # proportion de la barre parcourue
            # converti en int si on obtient un float
            progress = int(i / total * progress_bar_width)

            # {'█' * progress:<{progress_bar_width}} signifie qu'on veut
            # aligné le █ à gauche dans le champ de largeur progress_bar_width
            progress_bar = f"|{'█' * progress:<{progress_bar_width}}|"
            progress_percentage = progress * 100 // progress_bar_width
            progress_info = f"{progress_percentage}%{progress_bar} {i}/{total}"
            # affiche la progression de la barre formatté comme dans le sujet
            # '\r' permet de overwrite la ligne recente de la progression
            print(f"{progress_info}", end="\r", flush=True)



X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])
mylr = MyLinearRegression([[1.], [1.], [1.], [1.], [1]])

# Example 0:
y_hat = mylr.predict_(X)
print(y_hat)
# Output: array([[8.], [48.], [323.]])

# Example 1:
print(mylr.loss_elem_(Y, y_hat))
# Output: array([[225.], [0.], [11025.]])

# Example 2:
print(mylr.loss_(Y, y_hat))
# Output: 1875.0

# Example 3:
mylr.alpha = 1.6e-4
mylr.max_iter = 200000
mylr.fit_(X, Y)
print(mylr.thetas)
# Output: array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])

# Example 4:
y_hat = mylr.predict_(X)
print(y_hat)
# Output: array([[23.417..], [47.489..], [218.065...]])

# Example 5:
print(mylr.loss_elem_(Y, y_hat))
# Output: array([[0.174..], [0.260..], [0.004..]])

# Example 6:
print(mylr.loss_(Y, y_hat))
# Output: 0.0732..