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

features_dict = {
    # Binary variables
    "BPMEDS": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "TOLDHI2": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "DIABETE3": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "HLTHPLN1": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "QLACTLM2": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "EXERANY2": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "ADDEPEV2": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "ASTHMA3": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "SMOKE100": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "CVDSTRK3": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "HAVARTH3": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "CHCCOPD1": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "SEX": {"type": "binary", "missing_values": [np.nan]},
    "BPHIGH4": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "USEEQUIP": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "BLIND": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "DECIDE": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "DIFFWALK": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "DIFFDRES": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "DIFFALON": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "ALCDAY5": {"type": "binary", "missing_values": [np.nan]},
    "CHECKUP1": {"type": "binary", "missing_values": [np.nan]},
    "STRENGTH": {"type": "binary", "missing_values": [np.nan]},
    "AVG_FRUITS_VEGS": {"type": "binary", "missing_values": [np.nan]},
    "_RFBING5": {"type": "binary", "missing_values": [9, np.nan]},
    "_AGE65YR": {"type": "binary", "missing_values": [np.nan]},
    "_RFDRHV5": {"type": "binary", "missing_values": [9, np.nan]},
    "_TOTINDA": {"type": "binary", "missing_values": [9, np.nan]},
    "_PAINDX1": {"type": "binary", "missing_values": [9, np.nan]},
    "_PASTRNG": {"type": "binary", "missing_values": [9, np.nan]},
    "_PASTAE1": {"type": "binary", "missing_values": [9, np.nan]},
    "_FLSHOT6": {"type": "binary", "missing_values": [9, np.nan]},
    "_PNEUMO2": {"type": "binary", "missing_values": [9, np.nan]},
    "_AIDTST3": {"type": "binary", "missing_values": [9, np.nan]},
    "_RFBMI5": {"type": "binary", "missing_values": [9, np.nan]},
    "_HISPANC": {"type": "binary", "missing_values": [9, np.nan]},
    "_CASTHM1": {"type": "binary", "missing_values": [9, np.nan]},
    "_LTASTH1": {"type": "binary", "missing_values": [9, np.nan]},
    "_RFCHOL": {"type": "binary", "missing_values": [9, np.nan]},
    "_RFHYPE5": {"type": "binary", "missing_values": [9, np.nan]},
    "_HCVU651": {"type": "binary", "missing_values": [9, np.nan]},
    "_RFHLTH": {"type": "binary", "missing_values": [9, np.nan]},
    "MEDCOST": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "BLOODCHO": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "ASTHNOW": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "CHCSCNCR": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "CHCCOPD1": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "CHCOCNCR": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "CHCKIDNY": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "VETERAN3": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "INTERNET": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "LMTJOIN3": {"type": "binary", "missing_values": [7, 9, np.nan]},
    "ARTHDIS2": {"type": "binary", "missing_values": [7, 9, np.nan]},
    # Categorical variable
    "_RACEGR3": {
        "type": "categorical",
        "missing_values": [9, np.nan],
        "categories": [1, 2, 3, 4, 5],
    },
    # Numeric variables
    #'_DRNKWEK': {'type': 'numeric', 'missing_values': [99900, np.nan], 'range': (1, 6)},
    "_PAREC1": {"type": "numeric", "missing_values": [9, np.nan], "range": (1, 6)},
    "_PA150R2": {"type": "numeric", "missing_values": [9, np.nan], "range": (1, 6)},
    "_PACAT1": {"type": "numeric", "missing_values": [9], "range": (1, 6)},
    "FC60_": {"type": "numeric", "missing_values": [99900], "range": (1, 6)},
    "MAXVO2_": {"type": "numeric", "missing_values": [99900], "range": (1, 6)},
    "EDUCA": {
        "type": "numeric",
        "missing_values": [9, np.nan],
        "range": (1, 6),
    },  # Education level
    "_INCOMG": {"type": "numeric", "missing_values": [9, np.nan], "range": (1, 6)},
    "BMI": {"type": "numeric", "missing_values": [7777, 9999, np.nan], "range": (1, 4)},
    "MENTHLTH": {
        "type": "numeric",
        "missing_values": [77, 99, np.nan],
        "map_value": {88: 0},
        "range": (1, 30),
    },
    "_AGEG5YR": {"type": "numeric", "missing_values": [14, np.nan], "range": (1, 13)},
    "PHYSHLTH": {
        "type": "numeric",
        "missing_values": [77, 99, np.nan],
        "map_value": {88: 0},
        "range": (1, 30),
    },
}


# ------------------- Functions for data preprocessing ------------------------------ #
def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def csv_to_array(csv_file, id=False):
    """
    Read csv file and store content into array structure
    """
    data = []
    ids = []
    with open(csv_file, newline="") as file_csv:
        csvreader = csv.reader(file_csv, delimiter=",")
        # First extract column names from data
        column_names = next(csvreader)
        # List for storing ids

        for row in csvreader:
            if id:  # for x_test return id column
                ids.append(row[0])

                # Append blanck values as NaN (before leading with them)
                # Do not include first column as it serves as identifier
                data.append(
                    [float(val) if val != "" else float("nan") for val in row[1:]]
                )

            else:  # for x_train, y_train

                data.append(
                    [float(val) if val != "" else float("nan") for val in row[1:]]
                )

        # For column names dont return ID feature
        column_names = column_names[1:]
        # Transfrom nested list into 2D array
        data = np.array(data)
        if id:  # if X_test return id instead of features
            column_names = np.array(ids, dtype=int)

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

    # Identify FN and FP
    FN_indices = (y_te == 1) & (y_pred == 0)  # False Negatives
    FP_indices = (y_te == 0) & (y_pred == 1)  # False Positives

    # Analyze feature distributions for FN and FP
    # analyze_feature_distribution(x_te, y_te, FN_indices, FP_indices)

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


def analyze_feature_distribution(X, y_true, FN_indices, FP_indices):
    """
    Plot feature distributions for False Negatives (FN) and False Positives (FP) to analyze patterns.
    """
    num_features = X.shape[1]

    for feature_idx in range(num_features):
        feature_values = X[:, feature_idx]

        # Extract values for FN and FP samples
        FN_values = feature_values[FN_indices]
        FP_values = feature_values[FP_indices]

        # Plot distribution for FN and FP
        plt.figure(figsize=(10, 4))
        plt.hist(FN_values, bins=20, alpha=0.5, color="blue", label="False Negatives")
        plt.hist(FP_values, bins=20, alpha=0.5, color="red", label="False Positives")

        plt.title(f"Feature {feature_idx} Distribution for FN and FP")
        plt.xlabel(f"Feature {feature_idx} Values")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()


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
                y,
                X,
                k_indices,
                k,
                initial_w,
                max_iters,
                gamma,
                reg_norm,
                batch_size,
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
    y,
    x,
    k_indices,
    k,
    initial_w,
    max_iters,
    gamma,
    reg_norm,
    batch_size,
    beta1,
    beta2,
    decay_rate,
    decay_steps,
    plot=True,
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
        feature_values = X[:, feature_idx].copy()

        # Check for missing values and separate them as an additional category
        missing_mask = np.isin(feature_values, info["missing_values"])
        # missing_percentage = (np.sum(missing_mask) / len(feature_values)) * 100
        # print(f"Feature '{feature}': {missing_percentage:.2f}% missing values.")

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


def X_preprocessing(x, features):
    # Modifying BPHIGH4 column
    index_BPHIGH4 = features.index("BPHIGH4")
    x[:, index_BPHIGH4] = np.where(x[:, index_BPHIGH4] == 4, 1, x[:, index_BPHIGH4])
    x[:, index_BPHIGH4] = np.where(
        (x[:, index_BPHIGH4] == 2) | (x[:, index_BPHIGH4] == 3), 0, x[:, index_BPHIGH4]
    )

    # Consistent scaling (wont use 'EXEROFIT1')
    columns_to_scale = ["ALCDAY5", "STRENGTH"]
    indices = [features.index(col) for col in columns_to_scale]
    # Select the columns
    selected_columns = x[:, indices]

    # Preprocessing special values
    #  777/999 -> np.nan
    selected_columns = np.where(
        np.isin(selected_columns, [777, 999]), np.nan, selected_columns
    )
    # 888->0
    selected_columns = np.where(selected_columns == 888, 0, selected_columns)

    # If value smaller than 200, days per week entry; set additional condition; so 0 not modified
    mask_dw = (selected_columns > 0) & (selected_columns < 200)
    # Convert to string, remove leading 1, back to float; and scale
    selected_columns[mask_dw] = (
        np.char.lstrip(selected_columns[mask_dw].astype(str), "1").astype(float)
    ) * 4.345

    # If value greater than 200, days per month
    mask_dm = selected_columns > 200
    # Convert to string, remove leading 1, back to float; and scale
    selected_columns[mask_dm] = np.char.lstrip(
        selected_columns[mask_dm].astype(str), "2"
    ).astype(float)

    # Binary conversion for STRENGTH and ALCDAY5
    # STRENGTH: 0 if > 0, otherwise 1
    strength_index = columns_to_scale.index("STRENGTH")
    # Perserve Nans
    selected_columns[:, strength_index] = np.where(
        np.isnan(selected_columns[:, strength_index]),
        np.nan,
        np.where(selected_columns[:, strength_index] > 0, 0, 1),
    )

    # ALCDAY5: 0 if > 0, otherwise 1
    alcday_index = columns_to_scale.index("ALCDAY5")
    selected_columns[:, alcday_index] = np.where(
        np.isnan(selected_columns[:, alcday_index]),
        np.nan,
        np.where(selected_columns[:, alcday_index] > 0, 0, 1),
    )

    # Update x_train with the transformed columns
    x[:, indices] = selected_columns

    # JOINPAIN
    # Binary conversion
    """
    joinpain_index = features.index('JOINPAIN')
    # Perserve Nans
    x_train[:, joinpain_index] = np.where(np.isnan(x_train[:, joinpain_index]), np.nan,
        np.where(x_train[:, joinpain_index] > 7, 1, 0))

    # Poorhlth Binary
    poorh_index = features.index('POORHLTH')
    x_train[:, poorh_index] = np.where(np.isin(x_train[:, poorh_index], [77, 99]), np.nan, x_train[:, poorh_index])
    # Need to add condition so 88 (None) -> 0
    # Preserve NaNs and apply binary transformation for CHECKUP
    x_train[:, poorh_index] = np.where(
        np.isnan(x_train[:, poorh_index]), 
        np.nan,  # Keep NaNs as they are
        np.where((x_train[:, poorh_index] < 31) & (x_train[:, poorh_index] > 2), 1, 0)  # Set to 1 if equal to 1, otherwise 0
    )

    """

    # CHECKUP Binary
    checkup_index = features.index("CHECKUP1")
    x[:, checkup_index] = np.where(
        np.isin(x[:, checkup_index], [7, 9]), np.nan, x[:, checkup_index]
    )
    # Preserve NaNs and apply binary transformation for CHECKUP
    # Values = None = 8 -> 0 (since different than 1)
    x[:, checkup_index] = np.where(
        np.isnan(x[:, checkup_index]),
        np.nan,  # Keep NaNs as they are
        np.where(x[:, checkup_index] == 1, 1, 0),  # Set to 1 if equal to 1, otherwise 0
    )

    # Creating BMI column
    # Need processing for 7777, 9999, np.nan
    # First convert entries = 7777,9999 -> np.nan

    # 1. From kgs to pounds ('WEIGTH2' column)
    index_WEIGHT2 = features.index("WEIGHT2")
    # Refused and not sure entries
    x[:, index_WEIGHT2] = np.where(
        np.isin(x[:, index_WEIGHT2], [7777, 9999]), np.nan, x[:, index_WEIGHT2]
    )

    # Apply to values greater than 9000
    mask = x[:, index_WEIGHT2] > 8999
    # Convert to string, strip the first character if it's '9', then convert back to float
    x[mask, index_WEIGHT2] = np.char.lstrip(
        x[mask, index_WEIGHT2].astype(str), "9"
    ).astype(float)
    # Convert from kg to pounds
    x[mask, index_WEIGHT2] *= 2.20462

    # 2. From cm to in ('HEIGHT3' column)
    index_HEIGHT3 = features.index("HEIGHT3")
    # Not sure, refused values
    x[:, index_HEIGHT3] = np.where(
        np.isin(x[:, index_HEIGHT3], [7777, 9999]), np.nan, x[:, index_HEIGHT3]
    )

    # i. cm -> in
    # Apply transformation to values greater than 9000
    mask_cm = x[:, index_HEIGHT3] > 8999
    # Convert to string, strip the first character '9', convert back to float, and convert from cm to feet
    x[mask_cm, index_HEIGHT3] = np.char.lstrip(
        x[mask_cm, index_HEIGHT3].astype(str), "9"
    ).astype(float)
    x[mask_cm, index_HEIGHT3] *= 0.393701  # Convert from cm to in

    # ii. ftin -> in
    mask_ftin = x[:, index_HEIGHT3] < 8999
    # Ex:  504 -> 5*12 + 4 = 64 inches
    x[mask_ftin, index_HEIGHT3] = (x[mask_ftin, index_HEIGHT3] // 100) * 12 + (
        x[mask_ftin, index_HEIGHT3] % 100
    )

    # Extract the height in inches and weight in pounds for BMI calculation
    height_in_inches = x[:, index_HEIGHT3]
    weight_in_pounds = x[:, index_WEIGHT2]

    # Calculate BMI using the formula
    bmi = (weight_in_pounds * 703) / (
        height_in_inches**2
    )  # multiplying by 703 (so metric units)

    # Define bins for bmi

    """

    bmi_bins = [0, 16, 24.9, 29.9, 250]  # Underweight, Normal, Overweight, Obese
    bmi_risk_values = [3, 0, 1, 2]   # 0: Normal, 1: Overweight, 2:Obese, 3: Underweight (ordered by risk)

    # Digitize BMI values, excluding NaNs
    bmi_category = np.where(np.isnan(bmi), np.nan, np.digitize(bmi, bins=bmi_bins, right=True))

    # Map digitized values to risk values, keeping NaNs as NaN
    bmi_numerical = np.array([
        bmi_risk_values[int(bin_idx) - 1] if not np.isnan(bin_idx) and 1 <= bin_idx <= len(bmi_risk_values) else np.nan
        for bin_idx in bmi_category
    ])
    """
    # Define BMI as binary
    bmi_binary = np.where((bmi >= 16) & (bmi < 24.9), 0, 1)

    # Add the BMI column to x_train
    x = np.column_stack((x, bmi_binary))

    # Add column name to feature list
    features.append("BMI")

    # Creating average fruit and vegetable consumption
    # Indices of the columns to average
    columns_to_average = ["FRUTDA1_", "VEGEDA1_", "GRENDAY_", "ORNGDAY_", "BEANDAY_"]
    indices = [features.index(col) for col in columns_to_average]

    # Select the columns
    selected_columns = x[:, indices]

    # Calculate the average while ignoring NaN values
    # np.nanmean computes the mean ignoring NaN values
    # If all values are missing in a row, np.nanmean will return NaN
    average_column = np.nanmean(selected_columns, axis=1)

    # Convert into binary
    binary_avg_fruit_veg = np.where(average_column > 0.6, 0, 1)

    # Add the new average column to x_train
    x = np.column_stack((x, binary_avg_fruit_veg))

    # Add the new feature name to the features list
    features.append("AVG_FRUITS_VEGS")

    #'_DRNKWEK': cap
    # index_DRNKWEK = features.index('_DRNKWEK')

    # Cap values at 100
    # x_train[:, index_DRNKWEK] = np.minimum(x_train[:, index_DRNKWEK], 100)

    index_MAXO = features.index("MAXVO2_")
    x[:, index_MAXO] = np.minimum(x[:, index_MAXO], 50)

    index_FC = features.index("FC60_")
    x[:, index_FC] = np.minimum(x[:, index_FC], 10)

    return x, features


def K_fold_cross_validation_f1_score(
    X,
    y,
    K,
    seed,
    max_iters,
    gamma,
    reg_norm,
    prob_threshold,
    batch_size,
    beta1,
    beta2,
    decay_rate,
    decay_steps,
):
    """
    Runs k-fold cross-validation for a given dataset, a list of learning rates, and other configurable parameters.

    max_iters : int
        Maximum number of iterations for optimization.

    gammas :

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

    F1_score_average = 0

    for k in tqdm(range(K)):
        F1_score = cross_validation_f1_score(
            y,
            X,
            k_indices,
            k,
            initial_w,
            max_iters=max_iters,
            gamma=gamma,
            reg_norm=reg_norm,
            prob_threshold=prob_threshold,
            batch_size=batch_size,
            beta1=beta1,
            beta2=beta2,
            decay_rate=decay_rate,
            decay_steps=decay_steps,
        )
        F1_score_average += F1_score

    F1_score_average /= K

    return F1_score_average


def cross_validation_f1_score(
    y,
    x,
    k_indices,
    k,
    initial_w,
    max_iters,
    gamma,
    reg_norm,
    prob_threshold,
    batch_size,
    beta1,
    beta2,
    decay_rate,
    decay_steps,
    plot=True,
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

    idx_te = k_indices[k]
    idx_tr = k_indices[
        np.arange(len(k_indices)) != k
    ].flatten()  # array with all indices in K_indices but kth
    y_te = y[idx_te]
    x_te = x[idx_te]
    y_tr = y[idx_tr]
    x_tr = x[idx_tr]

    # ridge regression:
    w, _, _ = reg_logistic_regression(
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
    y_prob_tr = sigmoid(x_te @ w)
    y_pred_tr = np.where(
        y_prob_tr < prob_threshold, 0, 1
    )  # if below threshold 0 otherwise 1
    accuracy, F1_score = F1_score_f(y_te, y_pred_tr, plot=False)
    return F1_score
