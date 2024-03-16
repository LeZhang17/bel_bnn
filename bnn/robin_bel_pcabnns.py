import os

import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from keras import Input, Model
from keras.layers import Dense

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras

# get cwd
cwd = os.getcwd()
# Load data
# data = pd.read_pickle('datags.pkl')
data = pd.read_pickle(os.path.join(cwd, "bnn", 'datags.pkl'))
# Set time steps
time_steps = 30
X = data[:, :time_steps]
y = data[:, time_steps:]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # split the data into training and test sets
x_test, x_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)  # split the test set into test and validation sets

# Print the number of samples in each set
print(f"Number of samples in the training set: {X_train.shape[0]}")
print(f"Number of samples in the test set: {X_test.shape[0]}")
print(f"Number of samples in the validation set: {x_val.shape[0]}")

# Print the dimensions of the data
print(f"Dimensions of the predictors: {X_train.shape[1]}")
# 30
print(f"Dimensions of the target: {y_train.shape[1]}")
# 270

# We'll use the validation set for early stopping

# Initialize scalers
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Fit the scalers
scaler_X.fit(X_train)
scaler_y.fit(y_train)

# Transform the data
X_train_scaled = scaler_X.transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
x_val_scaled = scaler_X.transform(x_val)

y_train_scaled = scaler_y.transform(y_train)
y_test_scaled = scaler_y.transform(y_test)
y_val_scaled = scaler_y.transform(y_val)

# Initialize PCA - keep 99% of the variance
pca_X = PCA(n_components=.99)
pca_Y = PCA(n_components=.99)

# Fit the PCA
pca_X.fit(X_train_scaled)
pca_Y.fit(y_train_scaled)

# Print how many components PCA will keep
print(f"Number of components to keep for the predictors: {pca_X.n_components_}")
# Number of components to keep for the predictors: 5
print(f"Number of components to keep for the target: {pca_Y.n_components_}")
# Number of components to keep for the target: 3

# Transform the data
X_train_pca = pca_X.transform(X_train_scaled)
X_test_pca = pca_X.transform(X_test_scaled)
x_val_pca = pca_X.transform(x_val_scaled)

y_train_pca = pca_Y.transform(y_train_scaled)
y_test_pca = pca_Y.transform(y_test_scaled)
y_val_pca = pca_Y.transform(y_val_scaled)

# Define the model
# First set the input, hidden and output dimensions
input_shape = X_train_pca.shape[1]
output_dim = y_train_pca.shape[1]
hidden_units = 64  # number of hidden units in the dense layer
num_components = 10  # number of components in the mixture
params_size = tfp.layers.MixtureNormal.params_size(num_components, output_dim)  # size of the parameters

inputs = Input(shape=input_shape, name="input")  # input layer
x = Dense(hidden_units, activation="relu")(inputs)  # hidden layer
x = Dense(params_size, activation=None, name="output")(x)  # dense layer with no activation
outputs = tfp.layers.MixtureNormal(num_components, output_dim)(x)  # output layer - mixture normal distribution

# Create the model
model = Model(inputs=inputs, outputs=outputs, name="model")


# Define the loss function
def nll(y_true, y_pred):
    """
    Negative log likelihood loss function
    :param y_true: true values
    :param y_pred: predicted values
    :return: negative log likelihood
    """
    return -y_pred.log_prob(y_true)


# Compile the model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=nll)

# Define an early stopping callback
early_stopping = tfk.callbacks.EarlyStopping(
    monitor="val_loss",  # monitor the validation loss
    patience=10,  # if the validation loss does not improve for 10 epochs, stop training
    verbose=1,  # make the output verbose
    restore_best_weights=True,  # restore the best weights
)

# Fit the model
history = model.fit(
    X_train_pca,
    y_train_pca,
    epochs=100,
    batch_size=32,
    validation_data=(x_val_pca, y_val_pca),
    callbacks=[early_stopping],
)

# Plot the training and validation loss
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Test Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Now sample from the model
# Pick a test sample
n_test = 161
x = X_test_pca[n_test]
y = y_test_pca[n_test]

# Number of samples to draw
n_samples = 1000

# Draw samples
samples = model(x[None, ...]).sample(n_samples)
# Print the shape of the samples
print(samples.shape)
# (1000, 1, 3)

# Reshape the samples
samples = samples[:, 0, :]

# Inverse transform the samples
# Undo pca
samples = pca_Y.inverse_transform(samples)
# Undo scaling
samples = scaler_y.inverse_transform(samples)

# Inverse transform the true value
# Undo pca
y_true = pca_Y.inverse_transform(y[None, ...])
# Undo scaling
y_true = scaler_y.inverse_transform(y_true)

# Plot the samples and the true value
plt.figure(figsize=(10, 6))
for i in range(n_samples):
    plt.plot(samples[i], color="black", alpha=0.1)
plt.plot(y_true[0], label="True", color="red", linewidth=4)
plt.title("Samples and True Value")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.title(f"Test Sample {n_test} Predictions with Mixture Density Network")
plt.savefig("mdn.png", dpi=300, bbox_inches='tight')
plt.show()


