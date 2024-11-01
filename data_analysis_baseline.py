import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import csv

# === Logistic Regression Functions ===


def sigmoid(z):
    """
    Computes the sigmoid function, used to map predictions to a probability between 0 and 1.

    Args:
        z (ndarray): Linear combination of inputs and weights.

    Returns:
        ndarray: Transformed values between 0 and 1.
    """
    return 1 / (1 + np.exp(-z))


def reg_logistic_regression(
    y_tr, x_tr, y_te, x_te, initial_w, max_iters, gamma, reg_norm, batch_size
):
    """
    Regularized logistic regression using SGD or full-batch gradient descent with L1 or L2 regularization.

    Args:
        y_tr (ndarray): Training labels.
        x_tr (ndarray): Training features.
        y_te (ndarray): Test labels.
        x_te (ndarray): Test features.
        initial_w (ndarray): Initial weights for the model.
        max_iters (int): Maximum number of iterations for optimization.
        gamma (float): Learning rate.
        reg_norm (str): Type of regularization ('l2' for ridge or 'l1' for lasso).
        batch_size (int): Size of mini-batches for SGD (default is full-batch gradient descent).

    Returns:
        tuple: Final weights, list of training losses, and list of test losses.
    """
    N_tr = np.shape(y_tr)[0]
    w = initial_w
    losses_train = []
    losses_test = []

    if batch_size is None:
        batch_size = N_tr  # Full batch if not provided

    for n_iter in range(max_iters):
        indices = np.random.permutation(N_tr)
        x_tr = x_tr[indices]
        y_tr = y_tr[indices]

        for i in range(0, N_tr, batch_size):
            x_batch = x_tr[i : i + batch_size]
            y_batch = y_tr[i : i + batch_size]
            y_pred = sigmoid(x_batch @ w)

            if reg_norm == "l2":
                reg_gradient = 2 * gamma * w
            elif reg_norm == "l1":
                reg_gradient = gamma * np.sign(w)
            else:
                reg_gradient = 0

            gradient = compute_gradient_log_loss(y_batch, y_pred, x_batch, batch_size)
            gradient += reg_gradient
            w -= gamma * gradient

        losses_test.append(compute_log_loss(y_te, sigmoid(x_te @ w)))
        losses_train.append(compute_log_loss(y_tr, sigmoid(x_tr @ w)))

    return w, losses_train, losses_test


def logistic_regression(y_tr, x_tr, y_te, x_te, initial_w, max_iters, gamma):
    """
    Basic logistic regression using gradient descent (GD) without regularization.

    Args:
        y_tr (ndarray): Training labels.
        x_tr (ndarray): Training features.
        y_te (ndarray): Test labels.
        x_te (ndarray): Test features.
        initial_w (ndarray): Initial weights for the model.
        max_iters (int): Maximum number of iterations for optimization.
        gamma (float): Learning rate.

    Returns:
        tuple: Final weights, list of training losses, and list of test losses.
    """
    N_tr = np.shape(y_tr)[0]
    w = initial_w
    losses_train = []
    losses_test = []

    for n_iter in range(max_iters):
        y_pred = sigmoid(x_tr @ w)
        losses_train.append(compute_log_loss(y_tr, y_pred))
        gradient = compute_gradient_log_loss(y_tr, y_pred, x_tr, N_tr)
        w -= gamma * gradient
        losses_test.append(compute_log_loss(y_te, sigmoid(x_te @ w)))

    return w, losses_train, losses_test


# === Loss Functions ===


def compute_log_loss(y, y_pred):
    """
    Calculates the log loss (binary cross-entropy) for a set of predictions.

    Args:
        y (ndarray): True labels.
        y_pred (ndarray): Predicted probabilities.

    Returns:
        float: Log loss.
    """
    N = np.shape(y)[0]
    loss = (
        np.sum(-(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))) / N
    )
    return loss


def compute_gradient_log_loss(y, y_pred, tx, N):
    """
    Computes the gradient of the log loss for logistic regression.

    Args:
        y (ndarray): True labels.
        y_pred (ndarray): Predicted probabilities.
        tx (ndarray): Feature matrix.
        N (int): Number of samples.

    Returns:
        ndarray: Gradient of log loss.
    """
    return tx.T @ (y_pred - y) / N


# === Data Preprocessing and Balancing Functions ===


def csv_to_array(csv_file):
    """
    Read csv file and store content into array structure
    """
    data = []
    with open(csv_file, newline="") as file_csv:
        csvreader = csv.reader(file_csv, delimiter=",")
        # First extract column names from data
        column_names = next(csvreader)
        for row in csvreader:
            # Append blanck values as NaN (before leading with them)
            # Do not include first column as it serves as identifier
            data.append([float(val) if val != "" else float("nan") for val in row[1:]])

        # Transfrom nested list into 2D array
        data = np.array(data)

        # For column names dont return ID feature
        column_names = column_names[1:]

    return data, column_names


def balance_dataset(X, y):
    """
    Balances the dataset by undersampling the majority class to match the minority class.

    Args:
        X (ndarray): Feature matrix.
        y (ndarray): Labels array.

    Returns:
        tuple: Balanced feature matrix and labels array.
    """
    X_0 = X[y == 0]
    y_0 = y[y == 0]
    X_1 = X[y == 1]
    y_1 = y[y == 1]

    n_samples = len(y_1)
    indices = np.random.choice(len(y_0), n_samples, replace=False)
    X_0_balanced = X_0[indices]
    y_0_balanced = y_0[indices]

    X_balanced = np.vstack((X_0_balanced, X_1))
    y_balanced = np.hstack((y_0_balanced, y_1))
    indices = np.random.permutation(len(y_balanced))
    return X_balanced[indices], y_balanced[indices]


def train_test_split(X, y, training_split):
    """
    Splits the dataset into training and test sets based on a specified training split ratio.

    Args:
        X (ndarray): Feature matrix.
        y (ndarray): Labels array.
        training_split (float): Proportion of data to be used for training (0 < training_split < 1).

    Returns:
        tuple: Training features, training labels, test features, and test labels.
    """
    N = np.shape(y)[0]
    random_index = np.random.permutation(N)
    N_train = int(training_split * N)
    training_index = random_index[:N_train]
    test_index = random_index[N_train:]
    x_tr = X[training_index]
    x_te = X[test_index]
    y_tr = y[training_index]
    y_te = y[test_index]
    return x_tr, y_tr, x_te, y_te


# === Cross-Validation Functions ===


def cross_validation(
    y, x, k_indices, k, initial_w, max_iters, gamma, reg_norm, batch_size
):
    """return the loss for a fold (only for one FOLD) corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        initial_w:  initial weights
    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    """

    # ***************************************************
    # get k'th subgroup in test, others in train
    # Once have k folds, separate k -1 into train and leave one for test
    # Ex: choose k for test and remaining for training

    idx_te = k_indices[k]
    idx_tr = k_indices[
        np.arange(len(k_indices)) != k
    ].flatten()  # array with all indices in K_indices but kth
    y_te = y[idx_te]
    x_te = x[idx_te]
    y_tr = y[idx_tr]
    x_tr = x[idx_tr]

    # ridge regression:
    w, losses_tr, losses_te = reg_logistic_regression(
        y_tr, x_tr, y_te, x_te, initial_w, max_iters, gamma, reg_norm, batch_size
    )
    # ***************************************************
    return losses_tr, losses_te, w


def k_fold_cross_validation(
    X, y, max_iters, gammas, K, seed, reg_norm="", batch_size=None
):
    """
    Runs k-fold cross-validation for a given dataset, evaluating different learning rates.

    Args:
        X (ndarray): Feature matrix.
        y (ndarray): Labels array.
        max_iters (int): Maximum number of iterations for optimization.
        gammas (list): List of learning rates to evaluate.
        K (int): Number of folds for cross-validation.
        seed (int): Random seed for reproducibility.
        reg_norm (str): Regularization type ('l2' for ridge, 'l1' for lasso).
        batch_size (int): Batch size for SGD.

    Returns:
        tuple: Best gamma, corresponding test loss, and sorted dictionary of all gamma test losses.
    """
    initial_w = np.random.rand(X.shape[1]) * 0.01
    k_indices = build_k_indices(y, K, seed)
    loss_tr_list = []
    loss_te_list = []

    for gamma in gammas:
        loss_tr = 0
        loss_te = 0
        for k in range(K):
            losses_tr, losses_te, w = cross_validation(
                y, X, k_indices, k, initial_w, max_iters, gamma, reg_norm, batch_size
            )
            loss_tr += losses_tr[max_iters - 1]
            loss_te += losses_te[max_iters - 1]

        loss_tr_list.append(loss_tr / K)
        loss_te_list.append(loss_te / K)

    gamma_mse_dict = {gammas[i]: loss_te_list[i] for i in range(len(gammas))}
    sorted_gamma_mse = dict(sorted(gamma_mse_dict.items(), key=lambda item: item[1]))
    best_gamma, best_mse = list(sorted_gamma_mse.items())[0]

    return best_gamma, best_mse, sorted_gamma_mse


def build_k_indices(y, k_fold, seed):
    """
    Creates k indices for k-fold cross-validation.

    Args:
        y (ndarray): Labels array.
        k_fold (int): Number of folds.
        seed (int): Random seed for reproducibility.

    Returns:
        ndarray: Array of indices for each fold.
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


# === Analysis and Results Functions ===


def F1_score_f(y_true, y_pred, plot=True):
    """
    Computes the F1 score and optionally plots the confusion matrix.

    Args:
        y_true (ndarray): True labels.
        y_pred (ndarray): Predicted labels.
        plot (bool): Whether to plot the confusion matrix (default True).

    Returns:
        float: F1 score.
    """
    TP = TN = FP = FN = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            TP += 1
        elif true == 0 and pred == 0:
            TN += 1
        elif true == 0 and pred == 1:
            FP += 1
        elif true == 1 and pred == 0:
            FN += 1

    if plot:
        cm = np.array([[TN, FP], [FN, TP]])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            linewidths=0.5,
            linecolor="black",
            vmin=0,
            vmax=np.max(cm),
            alpha=0.7,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    return (2 * TP) / (2 * TP + FP + FN)
