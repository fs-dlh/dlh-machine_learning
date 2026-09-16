#!/usr/bin/env python3
""" Defines a Neuron class with FP using sigmoid activation. """
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """ Defines a single neuron performing binary classification.

    Private instance attributes:
        __W: weights vector (random normal)
        __b: bias (0)
        __A: activated output (0)
    """

    def __init__(self, nx):
        """ Constructor for the Neuron.

        Args:
            nx : Number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Getter for the weights vector. """
        return self.__W

    @property
    def b(self):
        """ Getter for the bias. """
        return self.__b

    @property
    def A(self):
        """ Getter for the activated output. """
        return self.__A

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neuron.

        Args:
            X : input data of shape (nx, m).

        Returns:
            numpy.ndarray: the activated output (sigmoid) of shape (1, m).
        """
        # Compute linear combination: z = W.X + b
        z = np.matmul(self.__W, X) + self.__b
        # Sigmoid activation
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """ Calculates the logistic regression cost.

        Args:
            Y : correct labels of shape (1, m).
            A : activated outputs of shape (1, m).

        Returns:
            float: the cost.
        """
        m = Y.shape[1]
        # Use 1.00000001 - A to avoid division by zero
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neuron's predictions and returns the cost.

        Args:
            X : input data of shape (nx, m).
            Y : correct labels of shape (1, m).

        Returns:
            tuple: (prediction, cost)
                prediction: numpy.ndarray of shape (1, m) with predicted labels
                            (1 if activation >= 0.5 else 0)
                cost: float
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = (A >= 0.5).astype(int)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Performs one pass of gradient descent on the neuron.

        Args:
            X : input data of shape (nx, m).
            Y : correct labels of shape (1, m).
            A : activated outputs of shape (1, m).
            alpha : learning rate.
        """
        m = X.shape[1]
        # Compute gradients
        dW = (1 / m) * np.matmul((A - Y), X.T)
        db = (1 / m) * np.sum(A - Y)
        # Update parameters
        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ Trains the neuron with optional verbose and graph outputs.

        Args:
            X : input data of shape (nx, m).
            Y : correct labels of shape (1, m).
            iterations : number of iterations to train.
            alpha : learning rate.
            verbose: if True, print cost at each step.
            graph: if True, plot training cost over iterations.
            step: interval for printing/plotting.

        Raises:
            TypeError: if iterations is not an integer or alpha is not a float.
            ValueError: if iterations or alpha are not positive.

        Returns:
            tuple: (prediction, cost) after training.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        self.forward_prop(X)
        cost_val = self.cost(Y, self.__A)

        cost_history = []
        iter_history = []

        cost_history.append(cost_val)
        iter_history.append(0)

        if verbose:
            print(f"Cost after 0 iterations: {cost_val}")

        for i in range(1, iterations + 1):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

            if i % step == 0 or i == iterations:
                current_cost = self.cost(Y, self.__A)
                cost_history.append(current_cost)
                iter_history.append(i)

                if verbose:
                    print(f"Cost after {i} iterations: {current_cost}")

        if graph:
            plt.plot(iter_history, cost_history, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
