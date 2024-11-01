# Import functions needed
from final_model import *


# Step 1: Data Loading
# --------------------
# Load training and testing data
X_train, features = csv_to_array("dataset/x_train.csv")
Y_train, labels = csv_to_array("dataset/y_train.csv")
X_test, ids = csv_to_array("dataset/x_test.csv", id=True)

# Flatten Y_train to convert it into a 1D array and set negative labels to zero
Y_train = Y_train.flatten()
Y_train = np.where(Y_train == -1, 0, Y_train)

# Step 2: Data Preprocessing
# --------------------------
# Preprocess training and test sets
X_train, features = X_preprocessing(X_train, features)
X_test, _ = X_preprocessing(X_test, features)

# Balance the dataset to handle class imbalance
X_train_balanced, Y_train_balanced = balance_dataset(X_train, Y_train)

# Transform features with one-hot encoding and median imputation
X_train_transformed = transform_features_column_median(
    X_train_balanced, features_dict, features
)
X_test_transformed = transform_features_column_median(X_test, features_dict, features)

# Step 3: Training and Validation Setup
# -------------------------------------
# Define model hyperparameters
max_iters = 1500
gamma = 0.001
beta1 = 0.9
beta2 = 0.97
reg_norm = ["l2", 0.001]
prob_threshold = 0.52
batch_size = 4096
decay_rate = 0.96
decay_steps = 100
plot = True

# Initialize weights
initial_w = np.zeros(np.shape(X_train_transformed)[1])

# Split data into training and validation sets (90% training, 10% validation)
N = Y_train_balanced.shape[0]
random_index = np.random.permutation(N)
N_train = int(0.9 * N)
training_index = random_index[:N_train]
validation_index = random_index[N_train:]

# Extract training and validation sets
x_tr = X_train_transformed[training_index]
y_tr = Y_train_balanced[training_index]
x_val = X_train_transformed[validation_index]
y_val = Y_train_balanced[validation_index]

# Step 4: Model Training
# ----------------------
# Train logistic regression model with regularization
w, losses_tr, losses_val = reg_logistic_regression(
    y_tr,
    x_tr,
    y_val,
    x_val,
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

# Step 5: Test Set Prediction
# ---------------------------
# Predict probabilities and labels for the test set
y_prob_test = sigmoid(X_test_transformed @ w)
y_pred_test = np.where(
    y_prob_test < prob_threshold, -1, 1
)  # Classify based on probability threshold

# Step 6: Model Validation
# ------------------------
# Predict on the validation set and compute F1 score and accuracy
y_prob_val = sigmoid(x_val @ w)
y_pred_val = np.where(
    y_prob_val < prob_threshold, 0, 1
)  # Classify based on probability threshold
accuracy, F1_score = F1_score_f(y_val, y_pred_val, plot=plot)

# Step 7: Submission File Creation
# --------------------------------
# Define the output filename and create submission CSV
submission_filename = "submission.csv"
create_csv_submission(ids, y_pred_test, submission_filename)
