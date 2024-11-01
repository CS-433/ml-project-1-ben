import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import csv
from implementations_ben import *
from itertools import combinations
import itertools
from collections import defaultdict

features = []
# ------------------- Functions for data preprocessing ------------------------------ #


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


def histogram_correlation(feature_name, x_train, Y_train, max_x):
    index = features.index(feature_name)

    # Balance the dataset
    X_train_balanced, Y_train_balanced = balance_dataset(x_train, Y_train)
    N = Y_train_balanced.shape[0]

    # Extract STRENGTH values and corresponding labels
    values = X_train_balanced[:, index]
    y_labels = Y_train_balanced

    # Define bins for STRENGTH
    bins = np.linspace(0, 30, 100)  # Adjusted to cover the expected range for STRENGTH

    # Calculate the average label per bin
    bin_indices = np.digitize(values, bins)
    average_labels = []
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if mask.any():
            avg_label = y_labels[mask].mean()  # Calculate average label for the bin
            average_labels.append(avg_label)
        else:
            average_labels.append(0)  # Default to 0 if no samples in bin

    # Plot the histogram with colored bins for STRENGTH
    plt.figure(figsize=(8, 6))
    for i in range(len(bins) - 1):
        bin_mask = bin_indices == i + 1
        bin_color = "orange" if average_labels[i] >= 0.5 else "blue"
        plt.hist(
            values[bin_mask],
            bins=[bins[i], bins[i + 1]],
            color=bin_color,
            edgecolor="black",
            alpha=0.7,
        )

    plt.xlabel(feature_name)
    plt.xlim(0, max_x)

    plt.ylabel("Frequency")
    plt.title(f"Distribution {feature_name} of with Average Label Color")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def analyze_correlation(
    features_dict,
    X,
    y,
    max_iters,
    gamma,
    reg_norm="",
    prob_threshold=0.5,
    batch_size=None,
    relative=True,
    feature_index=0,
):

    X = transform_features_column_median(X, features_dict, features)

    initial_w = np.random.rand(np.shape(X)[1]) * (0)  # Change weights
    N = np.shape(y)[0]  # number of samples
    random_index = np.random.permutation(N)
    # Proportion trainig: 0.9; test: 0.1
    N_train = int(0.9 * N)
    training_index = random_index[:N_train]
    test_index = random_index[N_train:]
    x_tr = X[training_index]
    x_te = X[test_index]
    y_tr = y[training_index]
    y_te = y[test_index]

    w, losses_tr, losses_te = reg_logistic_regression(
        y_tr, x_tr, y_te, x_te, initial_w, max_iters, gamma, reg_norm, batch_size
    )

    plot_prediction_vs_feature(
        x_te, y_te, w, feature_index=feature_index, relative=relative
    )
    # ***************************************************


def transform_features_no_missing(X, features_dict, original_features):
    """
    Transforms the features in X based on the encoding specified in features_dict,
    assuming there are no missing values.

    Args:
    - X: The input NumPy array (original dataset).
    - features_dict: Dictionary defining feature types and encoding.
    - original_features: List of feature names in the original order of X.

    Returns:
    - X_transformed: NumPy array with transformed features in the specified format.
    """
    transformed_features = []  # List to store transformed feature arrays

    for feature, info in features_dict.items():
        feature_idx = original_features.index(feature)
        feature_values = X[:, feature_idx]

        if info["type"] == "binary":
            # Binary variables: map Yes=1 -> 1 and No=2 -> 0
            feature_transformed = np.where(feature_values == 1, 1, 0)
            transformed_features.append(feature_transformed.reshape(-1, 1))

        elif info["type"] == "categorical":
            # One-hot encode categorical features, with a column for each category
            categories = info["categories"]
            for cat in categories:
                # Create a binary column for each category value
                category_mask = (feature_values == cat).astype(int)
                transformed_features.append(category_mask.reshape(-1, 1))

        elif info["type"] == "numeric":
            # Handle special values (e.g., 88 as 0 in MENTHLTH)
            if "map_value" in info:
                special_values = info["map_value"]
                for special, replacement in special_values.items():
                    feature_values = np.where(
                        feature_values == special, replacement, feature_values
                    )

            # Directly append the numeric feature after applying special value transformation
            transformed_features.append(feature_values.reshape(-1, 1))

    # Concatenate all transformed features to form the final transformed dataset
    X_transformed = np.hstack(transformed_features)
    return X_transformed


def marginal_contribution_F1_score(results):

    # Initialize dictionaries to store feature contributions
    feature_marginal_contrib = defaultdict(float)
    feature_counts = defaultdict(int)

    # Calculate the overall mean F1 score across all results
    mean_f1_score = sum(result["F1_score"] for result in results) / len(results)

    # Calculate marginal contributions for each feature across all results
    for result in results:  # Analyze all combinations
        f1_score = result["F1_score"]
        for feature in result["features"]:
            feature_counts[feature] += 1
            # Calculate marginal contribution (F1 score difference from the mean)
            feature_marginal_contrib[feature] += f1_score - mean_f1_score

    # Calculate average marginal contribution per feature
    feature_avg_marginal_contrib = {
        feature: feature_marginal_contrib[feature] / feature_counts[feature]
        for feature in feature_counts
    }

    # Sort features by their average marginal contributions (descending) for impact analysis
    sorted_features = sorted(
        feature_avg_marginal_contrib.items(), key=lambda x: x[1], reverse=True
    )

    # Display the analysis
    print("Feature Impact Analysis (All Results):")
    for feature, avg_marginal_contrib in sorted_features:
        print(
            f"Feature: {feature}, Average Marginal Contribution: {avg_marginal_contrib:.4f}"
        )


def evaluate_feature_combinations_with_mandatory(
    features_dict,
    X_train_balanced,
    Y_train_balanced,
    r,
    mandatory_features,
    max_iters,
    gamma,
    beta1,
    beta2,
    reg_norm,
    prob_threshold,
    batch_size,
    decay_rate,
    decay_steps,
    plot,
):
    """
    Evaluate all combinations of `r` features from `features_dict` while always including `mandatory_features`.

    Args:
    - features_dict (dict): Dictionary defining features and their configurations.
    - X_train_balanced (np.ndarray): Balanced training dataset.
    - Y_train_balanced (np.ndarray): Balanced training labels.
    - r (int): Total number of features in each combination, including mandatory features.
    - mandatory_features (list): List of feature names that must always be included.

    Returns:
    - List of dictionaries with each entry containing:
        - 'features': Tuple of feature names used in the combination.
        - 'F1_score': F1 score for the combination.
    """
    # List to store the results
    results = []

    # Separate mandatory features from the rest
    all_features = list(features_dict.keys())
    remaining_features = [f for f in all_features if f not in mandatory_features]

    # Calculate number of additional features to select (r - len(mandatory_features))
    additional_r = r - len(mandatory_features)

    # Generate combinations of the remaining features
    feature_combinations = list(combinations(remaining_features, additional_r))

    # Loop through each feature combination
    # for i in tqdm(range(len(feature_combinations))):
    for i in range(len(feature_combinations)):
        # Create the full feature combination by adding mandatory features
        feature_combination = mandatory_features + list(feature_combinations[i])

        # Select features for the current combination
        selected_features_dict = {
            feature: features_dict[feature] for feature in feature_combination
        }
        # Transform features for this combination
        X_train_transformed = transform_features_column_median(
            X_train_balanced, selected_features_dict, features
        )

        # Run logistic regression
        if plot:
            print(f"features:  {tuple(feature_combination)}")
        accuracy, F1_score = one_run_logistic_regression(
            X_train_transformed,
            Y_train_balanced,
            max_iters=max_iters,
            gamma=gamma,
            beta1=beta1,
            beta2=beta2,
            reg_norm=reg_norm,
            prob_threshold=prob_threshold,
            batch_size=batch_size,
            decay_rate=decay_rate,
            decay_steps=decay_steps,
            plot=plot,
        )

        # Store the feature combination and F1 score
        results.append(
            {
                "features": tuple(feature_combination),
                "Accuracy": accuracy,
                "F1_score": F1_score,
            }
        )

    # Sort results by accuracy in descending order to see best combinations at the top
    results = sorted(results, key=lambda x: x["Accuracy"], reverse=True)

    # Return sorted results
    return results


def balance_dataset(X, y):

    X_0 = X[y == 0]  # Samples with label 0
    y_0 = y[y == 0]
    X_1 = X[y == 1]  # Samples with label 1
    y_1 = y[y == 1]

    # Randomly sample from the majority class (label 0) to match the minority class (label 1)
    n_samples = len(y_1)  # Number of samples in the minority class
    indices = np.random.choice(len(y_0), n_samples, replace=False)
    X_0_balanced = X_0[indices]
    y_0_balanced = y_0[indices]

    # Combine the balanced datasets
    X_balanced = np.vstack((X_0_balanced, X_1))
    y_balanced = np.hstack((y_0_balanced, y_1))

    # Shuffle the balanced dataset
    indices = np.random.permutation(len(y_balanced))
    X_balanced = X_balanced[indices]
    y_balanced = y_balanced[indices]

    return X_balanced, y_balanced


def one_run_logistic_regression(
    X,
    y,
    max_iters,
    gamma,
    beta1=0.9,
    beta2=0.999,
    reg_norm=[""],
    prob_threshold=0.5,
    batch_size=None,
    decay_rate=0.96,
    decay_steps=100,
    plot=True,
):
    initial_w = np.random.rand(np.shape(X)[1]) * 0
    N = np.shape(y)[0]  # number of samples
    random_index = np.random.permutation(N)
    # Proportion trainig: 0.9; test: 0.1
    N_train = int(0.9 * N)
    training_index = random_index[:N_train]
    test_index = random_index[N_train:]
    x_tr = X[training_index]
    x_te = X[test_index]
    y_tr = y[training_index]
    y_te = y[test_index]

    w, losses_tr, losses_te = reg_logistic_regression(
        y_tr,
        x_tr,
        y_te,
        x_te,
        initial_w,
        max_iters=max_iters,
        gamma=gamma,
        reg_norm=reg_norm,
        batch_size=batch_size,
        beta1=beta1,
        beta2=beta2,
        decay_rate=decay_rate,
        decay_steps=decay_steps,
    )

    y_prob = sigmoid(x_te @ w)
    y_pred = np.where(y_prob < prob_threshold, 0, 1)  # if below threshold 0 otherwise 1
    accuracy, F1_score = F1_score_f(y_te, y_pred, plot=plot)

    if plot:
        plot_roc_curve(y_te, y_prob)
        plot_loss_iter(
            losses_train=losses_tr,
            losses_test=losses_te,
            max_iters=max_iters,
            gamma=gamma,
        )
        plot_prediction_vs_feature(x_te, y_te, w, feature_index=0)
    # ***************************************************
    return accuracy, F1_score


import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot_prediction_vs_feature(X, y, w, feature_index=0, relative=True):
    """
    Plots the predicted probability as a function of a single feature value along with the decision boundary.
    Additionally, shows the distribution of actual labels (0s and 1s) for each feature value using stacked bars.

    Args:
    - X: The input NumPy array (features).
    - y: The actual target values.
    - w: The weight vector from the logistic regression model.
    - feature_index: Index of the feature to plot against the predictions (default is 0).
    - relative: Whether to plot relative (normalized) or absolute counts for the histogram.
    """
    # Extract the chosen feature values
    feature_values = X[:, feature_index]

    # Calculate the predicted probability for each feature value
    z = (
        feature_values * w[feature_index] + w[-1]
        if len(w) > 1
        else feature_values * w[feature_index]
    )
    y_pred = sigmoid(z)

    # Create a smooth range of feature values for the prediction line and decision boundary
    feature_range = np.linspace(feature_values.min(), feature_values.max(), 100)
    z_range = (
        feature_range * w[feature_index] + w[-1]
        if len(w) > 1
        else feature_range * w[feature_index]
    )
    y_pred_range = sigmoid(z_range)

    # Count occurrences of each class (0 and 1) for each unique feature value
    unique_values = np.unique(feature_values)
    counts_0 = [(feature_values[y == 0] == value).sum() for value in unique_values]
    counts_1 = [(feature_values[y == 1] == value).sum() for value in unique_values]
    total_counts = [counts_0[i] + counts_1[i] for i in range(len(unique_values))]

    # Choose between normalized and absolute values based on `relative`
    if relative:
        plot_counts_0 = [
            counts_0[i] / total_counts[i] if total_counts[i] > 0 else 0
            for i in range(len(unique_values))
        ]
        plot_counts_1 = [
            counts_1[i] / total_counts[i] if total_counts[i] > 0 else 0
            for i in range(len(unique_values))
        ]
        y_label = "Relative Frequency of 0s and 1s"
    else:
        plot_counts_0 = counts_0
        plot_counts_1 = counts_1
        y_label = "Absolute Count of 0s and 1s"

    # Set up the plot with a secondary y-axis for the probability line
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Primary y-axis (left) for counts
    ax1.bar(unique_values, plot_counts_0, color="blue", label="Actual 0", alpha=0.7)
    ax1.bar(
        unique_values,
        plot_counts_1,
        bottom=plot_counts_0,
        color="orange",
        label="Actual 1",
        alpha=0.7,
    )
    ax1.set_xlabel(f"Feature {feature_index} Value")
    ax1.set_ylabel(y_label)
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Secondary y-axis (right) for predicted probability
    ax2 = ax1.twinx()
    ax2.plot(
        feature_range,
        y_pred_range,
        color="green",
        linestyle="-",
        label="Predicted probability",
    )
    ax2.axhline(0.5, color="red", linestyle="--", label="Decision boundary")
    ax2.set_ylabel("Predicted Probability")
    ax2.set_ylim(0, 1)  # Set y-axis limits for probability between 0 and 1
    ax2.legend(loc="upper right")

    plt.title("Predicted Probability and Actual Label Distribution")
    plt.show()


def k_fold_cross_validation(
    X, y, max_iters, gammas, K, seed, reg_norm=[""], batch_size=None
):
    """
    Runs k-fold cross-validation for a given dataset, a list of learning rates, and other configurable parameters.

    max_iters : int
        Maximum number of iterations for optimization.

    gammas : array-like
        A list of learning rates (gammas) to evaluate.

    K : int
        The number of folds for cross-validation.

    seed : int
        Random seed to ensure reproducibility of the k-fold splits.

    reg_norm: norm: 'l2' for ridge (L2 norm) or 'l1' for lasso (L1 norm) regularization; by default none
    """

    # Initial weights (small random values)
    initial_w = np.random.rand(X.shape[1]) * 0.01
    # Split data into k folds
    k_indices = build_k_indices(y, K, seed)

    # Define lists to store the loss of training data and test data
    loss_tr_list = []
    loss_te_list = []

    # For each gamma, perform cross-validation
    for gamma in gammas:
        loss_tr = 0
        loss_te = 0
        for k in tqdm(range(K)):
            losses_tr, losses_te, w = cross_validation(
                y, X, k_indices, k, initial_w, max_iters, gamma, reg_norm, batch_size
            )

            if k == 0:
                plot_loss_iter(
                    losses_train=losses_tr,
                    losses_test=losses_te,
                    max_iters=max_iters,
                    gamma=gamma,
                )

            # Accumulate the last loss of the training and test sets
            loss_tr += losses_tr[max_iters - 1]
            loss_te += losses_te[max_iters - 1]

        # Average over all k folds
        loss_tr_list.append(loss_tr / K)
        loss_te_list.append(loss_te / K)

        print(
            f"Learning rate = {gamma}; Average train loss = {loss_tr / K}; Average test loss: {loss_te / K}"
        )

    # Create a dictionary that pairs gamma with the corresponding test loss (MSE)
    gamma_mse_dict = {gammas[i]: loss_te_list[i] for i in range(len(gammas))}

    # Sort the dictionary by test loss (MSE)
    sorted_gamma_mse = dict(sorted(gamma_mse_dict.items(), key=lambda item: item[1]))

    # Retrieve the best gamma (learning rate) with the lowest MSE
    best_gamma, best_mse = list(sorted_gamma_mse.items())[0]

    print(f"Best learning rate: {best_gamma} with test MSE: {best_mse}")

    return best_gamma, best_mse, sorted_gamma_mse


def train_test_split(X, y, training_split):
    """
    Splits the dataset into training and test sets based on a training split ratio.
    """
    N = np.shape(y)[0]  # number of samples
    random_index = np.random.permutation(N)
    # Proportion trainig: 0.9; test: 0.1´
    N_train = int(training_split * N)
    training_index = random_index[:N_train]
    test_index = random_index[N_train:]
    x_tr = X[training_index]
    x_te = X[test_index]
    y_tr = y[training_index]
    y_te = y[test_index]

    return x_tr, y_tr, x_te, y_te


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
        # For each row one fold, where elements are elements per fold (indicated by indices)

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]  # number of samples
    interval = int(num_row / k_fold)  # number of samples per row
    np.random.seed(seed)
    indices = np.random.permutation(
        num_row
    )  # create array with N elements from 0...N, shuffled

    # Group indices into arrays: do it K times ( so K folds)
    # take indices[0:i]; where i num of samples per fold; iteratively
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(
    y, x, k_indices, k, initial_w, max_iters, gamma, reg_norm, batch_size, beta1, beta2
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

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527597114, 0.33555914361295175)
    """

    # ***************************************************
    # INSERT YOUR CODE HERE
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
        y_tr,
        x_tr,
        y_te,
        x_te,
        initial_w,
        max_iters,
        gamma,
        reg_norm,
        batch_size,
        beta1,
        beta2,
    )
    # ***************************************************
    return losses_tr, losses_te, w


def reg_logistic_regression(
    y_tr,
    x_tr,
    y_te,
    x_te,
    initial_w,
    max_iters,
    gamma,
    reg_norm,
    batch_size,
    beta1=0.9,
    beta2=0.999,
    decay_rate=0.96,
    decay_steps=100,
    epsilon=1e-8,
):
    """
    Regularized logistic regression using Adam optimizer with L1 or L2 regularization and learning rate scheduler.

    - reg_norm: ('l2', weight) for ridge (L2 norm) or ('l1', weight) for lasso (L1 norm) regularization.
    - batch_size: number of samples to use for each gradient update in SGD. Default is full-batch gradient descent.
    - decay_rate: rate of exponential decay for the learning rate.
    - decay_steps: how often to apply decay (in iterations).
    """
    N_tr = np.shape(y_tr)[0]
    w = initial_w
    losses_train = []
    losses_test = []

    # Adam optimizer variables
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    t = 0  # time step

    # Set batch_size to N_tr if not provided (full gradient descent by default)
    if batch_size is None:
        batch_size = N_tr

    for n_iter in range(max_iters):
        # Apply learning rate decay
        if n_iter % decay_steps == 0 and n_iter > 0:
            gamma = gamma * decay_rate  # Exponential decay

        # Shuffle the training data at each iteration
        indices = np.random.permutation(N_tr)
        x_tr = x_tr[indices]
        y_tr = y_tr[indices]

        # Perform updates in mini-batches
        for i in range(0, N_tr, batch_size):
            x_batch = x_tr[i : i + batch_size]
            y_batch = y_tr[i : i + batch_size]

            # Prediction for the current batch
            y_pred = sigmoid(x_batch @ w)

            # Regularization terms
            if reg_norm[0] == "l2":
                # L2 regularization term (ridge)
                reg_gradient = reg_norm[1] * gamma * w  # L2 gradient
            elif reg_norm[0] == "l1":
                # L1 regularization term (lasso)
                reg_gradient = gamma * np.sign(w)  # L1 gradient
            else:  # No regularization
                reg_gradient = 0

            # Compute gradient
            gradient = compute_gradient_log_loss(y_batch, y_pred, x_batch, batch_size)

            # Update gradient with regularization
            gradient += reg_gradient

            # Increment time step for Adam
            t += 1

            # Update biased first and second moment estimates
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient**2)

            # Compute bias-corrected moment estimates
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            # Update weights using Adam update rule
            w = w - gamma * m_hat / (np.sqrt(v_hat) + epsilon)

        # Compute loss for test and train data (without adding regularization term)
        losses_test.append(compute_log_loss(y_te, sigmoid(x_te @ w)))
        losses_train.append(compute_log_loss(y_tr, sigmoid(x_tr @ w)))

    return w, losses_train, losses_test


'''

def reg_logistic_regression(y_tr, x_tr, y_te, x_te, initial_w, max_iters, gamma, reg_norm, batch_size):
    """
    Regularized logistic regression using SGD or full-batch gradient descent with L1 or L2 regularization.
    
    - reg_norm: 'l2' for ridge (L2 norm) or 'l1' for lasso (L1 norm) regularization.
    - batch_size: number of samples to use for each gradient update in SGD. Default is full-batch gradient descent.
    """
    N_tr = np.shape(y_tr)[0]
    w = initial_w
    losses_train = []
    losses_test = []
    
    # Set batch_size to N_tr if not provided (full gradient descent by default)
    if batch_size is None:
        batch_size = N_tr
    
    for n_iter in range(max_iters):
        # Shuffle the training data at each iteration
        indices = np.random.permutation(N_tr)
        x_tr = x_tr[indices]
        y_tr = y_tr[indices]

        # Perform updates in mini-batches
        for i in range(0, N_tr, batch_size):
            x_batch = x_tr[i:i + batch_size]
            y_batch = y_tr[i:i + batch_size]

            # Prediction for the current batch
            y_pred = sigmoid(x_batch @ w)

            # Regularization terms
            if reg_norm == 'l2':
                # L2 regularization term (ridge)
                reg_gradient = 2 * gamma * w  # L2 gradient
            elif reg_norm == 'l1':
                # L1 regularization term (lasso)
                reg_gradient = gamma * np.sign(w)  # L1 gradient
            else:  # No regularization
                reg_gradient = 0

            # Compute gradient
            gradient = compute_gradient_log_loss(y_batch, y_pred, x_batch, batch_size)

            # Update gradient with regularization
            gradient += reg_gradient

            # Update weights using the current batch's gradient
            w = w - gamma * gradient
            #print(w)

        # Compute loss for test and train data (without adding regularization term)
        losses_test.append(compute_log_loss(y_te, sigmoid(x_te @ w)))
        losses_train.append(compute_log_loss(y_tr, sigmoid(x_tr @ w)))

    return w, losses_train, losses_test
'''


def logistic_regression(y_tr, x_tr, y_te, x_te, initial_w, max_iters, gamma):
    """
    Logistic regression using GD; keeping track of loss at each iteration
    Logistic regression: predicts probability of belonging to binary class -> 0,1
                         The linear combination of x and w goes through sigmoid -> 0/1
    However, loss function and gradient now follow different expressions (than for conventional linear regression)
    """
    N_tr = np.shape(y_tr)[0]
    w = initial_w
    losses_train = []
    losses_test = []
    # Augmenting data

    for n_iter in range(max_iters):
        # Prediction
        y_pred = sigmoid(x_tr @ w)
        # Return loss at last iteration
        losses_train.append(compute_log_loss(y_tr, y_pred))
        # Compute gradient
        gradient = compute_gradient_log_loss(y_tr, y_pred, x_tr, N_tr)
        # Update w by gradient
        w = w - gamma * gradient
        # Compute loss for test data
        losses_test.append(compute_log_loss(y_te, sigmoid(x_te @ w)))
    return w, losses_train, losses_test


# Additional methods for implementations


def compute_log_loss(y, y_pred):
    """Calculate the loss according to log loss
    Args:
        y: numpy array of shape=(N, )
        y_pred: prediction vector of shape= (N,)

    Returns:
        the value of the loss (a scalar), corresponding to the prediction vector.
    """
    # Add small epsilon to log to avoid overflow in case of log(0)
    # Use * for element-wise multiplication
    N = np.shape(y)[0]
    loss = (
        np.sum(-(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))) / N
    )
    return loss


def sigmoid(z):
    """
    Implementation of sigmoid function for logistic regression
    Arg:
       z: 1D array of N
    Return:
       1D array with N entries with values between 0 and 1
    """
    return 1 / (1 + np.exp(-z))


def compute_gradient_log_loss(y, y_pred, tx, N):
    gradient = tx.T @ (y_pred - y) / N  # Normalize by N
    return gradient


def compute_MSE_loss(error, N, loss_type="MSE"):
    """Calculate the loss using either MSE (default), or RMSE.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    loss = np.sum((error) ** 2) / (2 * N)
    if loss_type == "RMSE":
        loss = np.sqrt(loss)
    return loss


def F1_score_f(y_true, y_pred, plot=True):
    """
    Compute the confusion matrix for binary classification and plot it; and return F1 score

    Args:
        y_true (array-like): True labels (0 or 1).
        y_pred (array-like): Predicted labels (0 or 1).

    Returns:
        tuple: (TP, TN, FP, FN) and plot of the confusion matrix.
    """
    # Initialize counts

    N = np.shape(y_true)[0]
    TP = TN = FP = FN = 0

    # Count TP, TN, FP, FN
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            TP += 1  # True Positive
        elif true == 0 and pred == 0:
            TN += 1  # True Negative
        elif true == 0 and pred == 1:
            FP += 1  # False Positive
        elif true == 1 and pred == 0:
            FN += 1  # False Negative

    if plot:

        # Create confusion matrix
        cm = np.array([[TN, FP], [FN, TP]])

        # Plotting
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            linewidths=0.5,
            linecolor="black",
            vmin=0,
            vmax=np.max(cm),  # Set saturation based on max value
            alpha=0.7,
        )  # Transparency for low values
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(ticks=[0.5, 1.5], labels=["0", "1"])
        plt.yticks(ticks=[0.5, 1.5], labels=["0", "1"])
        plt.show()

    # Calculate accuracy and F1 score with checks to avoid division by zero
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    F1_score = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

    return accuracy, F1_score


def plot_loss_iter(losses_train, losses_test, max_iters, gamma):
    plt.figure(figsize=(6, 4))
    plt.loglog(
        range(max_iters), losses_train, label="Training Loss", color="blue", marker="o"
    )
    plt.loglog(
        range(max_iters), losses_test, label="Test Loss", color="red", marker="x"
    )
    plt.title(f"Learning rate: {gamma}")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_roc_curve(y_true, y_prob):
    """
    Plots the ROC curve manually.

    Args:
    - y_true: True binary labels (0 or 1).
    - y_prob: Predicted probabilities for the positive class.
    """
    thresholds = np.linspace(0, 1, 100)  # Define a range of thresholds
    tprs = []  # True Positive Rates
    fprs = []  # False Positive Rates

    # Loop through each threshold and calculate TPR and FPR
    for threshold in thresholds:
        # Make predictions based on the threshold
        y_pred = (y_prob >= threshold).astype(int)

        # Calculate True Positives, False Positives, True Negatives, and False Negatives
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))

        # Calculate TPR and FPR
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensitivity or Recall
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # Fall-out

        tprs.append(TPR)
        fprs.append(FPR)

    # Plot ROC curve
    plt.figure(figsize=(6, 4))
    plt.plot(fprs, tprs, marker="o", label="ROC Curve")
    plt.plot(
        [0, 1], [0, 1], "k--", label="Random Guess"
    )  # Diagonal line for random guess
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


def transform_features_column_for_missing(X, features_dict, original_features):
    """
    Transforms the features in X using one-hot encoding, treating missing values as a separate category.

    Args:
    - X: The input NumPy array (original dataset).
    - features_dict: Dictionary defining feature types and encoding.
    - original_features: List of feature names in the original order of X.

    Returns:
    - X_transformed: NumPy array with transformed features in the specified format.
    """
    transformed_features = []  # List to store transformed feature arrays

    for feature, info in features_dict.items():
        feature_idx = original_features.index(feature)
        feature_values = X[:, feature_idx]

        # Check for missing values and separate them as an additional category
        # Mask for all samples with missing values for feature
        missing_mask = np.isin(feature_values, info["missing_values"])

        if info["type"] == "binary":
            # Binary variable: 1 if "Yes", 0 if "No", and extra column for missing values
            feature_transformed = np.where(feature_values == 1, 1, 0)
            feature_transformed[missing_mask] = (
                0  # Assign missing values to 0 in binary encoding col
            )
            transformed_features.append(feature_transformed.reshape(-1, 1))

            # Add an extra column for missing values
            transformed_features.append(missing_mask.astype(int).reshape(-1, 1))

        elif info["type"] == "categorical":
            # One-hot encode categorical features, adding a column for each category
            categories = info["categories"]
            for cat in categories:
                category_mask = (feature_values == cat).astype(int)
                transformed_features.append(category_mask.reshape(-1, 1))

            # Add an extra column for missing values
            transformed_features.append(missing_mask.astype(int).reshape(-1, 1))

        elif info["type"] == "numeric":

            # Handle special values (e.g., map 88 as 0 in MENTHLTH)
            if "map_value" in info:
                special_values = info["map_value"]
                for special, replacement in special_values.items():
                    feature_values = np.where(
                        feature_values == special, replacement, feature_values
                    )

            # Replace missing values with 0 or other placeholder, if necessary
            feature_transformed = feature_values.astype(float)
            feature_transformed[missing_mask] = (
                0  # Placeholder for missing numeric values
            )
            transformed_features.append(feature_transformed.reshape(-1, 1))

            # Add an extra column for missing values
            transformed_features.append(missing_mask.astype(int).reshape(-1, 1))

    # Concatenate all transformed features to form the final transformed dataset
    X_transformed = np.hstack(transformed_features)
    return X_transformed


def transform_features_column_median(X, features_dict, original_features):
    """
    Transforms the features in X using one-hot encoding and median imputation for numeric features

    Args:
    - X: The input NumPy array (original dataset).
    - features_dict: Dictionary defining feature types and encoding.
    - original_features: List of feature names in the original order of X.

    Returns:
    - X_transformed: NumPy array with transformed features in the specified format.
    """
    transformed_features = []  # List to store transformed feature arrays

    for feature, info in features_dict.items():
        feature_idx = original_features.index(feature)
        feature_values = X[:, feature_idx]

        # Check for missing values and separate them as an additional category
        missing_mask = np.isin(feature_values, info["missing_values"])

        if info["type"] == "binary":
            # Binary variable: 1 if "Yes", 0 if "No", and extra column for missing values
            feature_transformed = np.where(feature_values == 1, 1, 0)
            feature_transformed[missing_mask] = (
                0  # Assign missing values to 0 in binary encoding column
            )
            transformed_features.append(feature_transformed.reshape(-1, 1))

            # Add an extra column for missing values
            transformed_features.append(missing_mask.astype(int).reshape(-1, 1))

        elif info["type"] == "categorical":
            # One-hot encode categorical features, adding a column for each category
            categories = info["categories"]
            for cat in categories:
                category_mask = (feature_values == cat).astype(int)
                transformed_features.append(category_mask.reshape(-1, 1))

            # Add an extra column for missing values
            transformed_features.append(missing_mask.astype(int).reshape(-1, 1))

        elif info["type"] == "numeric":
            # Handle special values (e.g., map 88 as 0 in MENTHLTH)
            if "map_value" in info:
                special_values = info["map_value"]
                for special, replacement in special_values.items():
                    feature_values = np.where(
                        feature_values == special, replacement, feature_values
                    )

            # Compute the median of the feature, excluding missing values
            median_value = np.nanmedian(feature_values[~missing_mask].astype(float))

            # Replace missing values with the computed median
            feature_transformed = feature_values.astype(float)
            feature_transformed[missing_mask] = (
                median_value  # Median imputation for missing values
            )
            transformed_features.append(feature_transformed.reshape(-1, 1))

    # Concatenate all transformed features to form the final transformed dataset
    X_transformed = np.hstack(transformed_features)
    return X_transformed
