"""
Step 2 of Project 1 of the ML course:
Implementation of 6 functions:
    -   mean_squared_error_gd
    -   mean_squared_error_sgd
    -   least_squared
    -   ridge_regression
    -   logistic_regression
    -   reg_ridge_regression
"""

import numpy as np

# ************************************
# mean_squared_error_gd
# ************************************


def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    loss = np.sum(np.pow(y - tx @ w, 2)) / (2 * y.shape[0])
    return loss


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    error = y - (tx @ w)
    return -(tx.T @ error) / y.shape[0]


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initialization for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the weight vector of the final iteration
        loss: a scalar denoting the loss of the final iteration
    """
    w = initial_w

    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient

    loss = compute_loss(y, tx, w)

    return w, loss


# ************************************
# mean_squared_error_sgd
# ************************************

## Alternative stochastic gradient descent


def compute_gradient_stochastic(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: scalar
        tx: numpy array of shape=(D, )
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    error = y - (tx @ w)
    return -tx.T * error


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm

    Args:
        y : numpy array of shape (N, )
        tx : numpy array of shape(N, D)
        initial_w : numpy array of shape(D,). Initialization of the model parameters
        max_iters : scalar for the total number of iterations of sgd
        gamma : scalar denoting stepsize

    Returns:
        w: the model parameters as numpy array of shape (D,) after the last iteration of sgd
        loss: the loss value (scalar) for the last iteration of sgd
    """
    w = initial_w

    for n_iter in range(max_iters):
        # take one random point for gradient computation
        i = np.random.randint(len(y))
        y_i = y[i]
        tx_i = tx[i]
        gradient = compute_gradient_stochastic(y_i, tx_i, w)
        w = w - gamma * gradient
    loss = compute_loss(y, tx, w)

    return w, loss


# ************************************
# least_squared
# ************************************


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: the loss value, scalar

    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)

    loss = compute_loss(y, tx, w)
    return w, loss


# ************************************
# ridge_regression
# ************************************


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss : the loss value (scalar)

    """
    N, D = tx.shape
    w = np.linalg.solve(tx.T @ tx + 2 * N * lambda_ * np.identity(D), tx.T @ y)
    loss = compute_loss(y, tx, w)
    return w, loss


def sigma(n):
    """
    The sigma function for logistic regression.

    Args:
    n: scalar or array

    Returns:
    scalar or array, result of the sigma function
    """
    e = np.exp(n)
    return e / (1 + e)


def compute_loss_logistic(y, tx, w):
    """
    The logistic regression loss function.

    Args:
    y: NumPy array of shape (N,). Labels
    tx: numpy array of shape (N, D+1?)
    w: numpy array of shape=(D+1?, ). The vector of model parameters.

    Returns:
    Scalar denoting the loss.
    """
    txw = tx @ w
    loss = np.mean(np.log(1 + np.exp(txw)) - y * txw)
    return loss


def compute_gradient_logistic(y, tx, w):
    """
    The gradient of the logistic loss.

    Args:
    y: NumPy array of shape (N,). Labels
    tx: NumPy array of shape (N, D+1?)
    w: NumPy array of shape=(D+1?, ). The vector of model parameters.

    Returns:
    A NumPy array of shape (D, ), containing the gradient of the loss at w.

    """
    s = sigma(tx @ w)
    gradient = tx.T @ (s - y) / len(y)
    return gradient


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Perform logistic regression using gradient descent.

    Parameters:
    y: Labels (vector of size N)
    tx: Feature matrix (N x D+1) where the first column is the bias (ones)
    initial_w: Initial weights (D+1,)
    max_iters: Number of iterations for gradient descent
    gamma: Learning rate (scalar)

    Returns:
    The final weights after gradient descent.
    """
    w = initial_w
    for i in range(max_iters):
        # Compute the gradient
        gradient = compute_gradient_logistic(y, tx, w)

        # Update the weights
        w = w - (gamma * gradient)

    loss = compute_loss_logistic(y, tx, w)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Perform regularized logistic regression using gradient descent.

    Parameters:
    y: Labels (vector of size N)
    tx: Feature matrix (N x D+1) where the first column is the bias (ones)
    initial_w: Initial weights (D+1,)
    max_iters: Number of iterations for gradient descent
    gamma: Learning rate (scalar)
    lambda_: Regularization term (scalar)

    Returns:
    The final weights after gradient descent and the final loss.
    """
    w = initial_w

    for n_iter in range(max_iters):
        # Compute the gradient
        gradient = compute_gradient_logistic(y, tx, w)
        gradient += 2 * lambda_ * w

        # Update the weights
        w = w - (gamma * gradient)
    loss = compute_loss_logistic(y, tx, w)
    return w, loss
