import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import properscoring as ps
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt
import seaborn as sns

from keras import Input, Model
from keras.layers import Dense

from numpy.random import default_rng

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras

# get cwd
cwd = os.getcwd()
# Load data
data = pd.read_pickle('datags.pkl')

# data = pd.read_pickle(os.path.join(cwd, "bnn", 'datags.pkl'))
# Set time steps
time_steps = 30
X = data[:, :time_steps]
y = data[:, time_steps:]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # split the data into training and test sets
x_test, x_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)  # split the test set into test and validation sets

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
x_test_scaled = scaler_X.transform(x_test)

y_train_scaled = scaler_y.transform(y_train)
y_test_scaled = scaler_y.transform(y_test)
y_val_scaled = scaler_y.transform(y_val)

X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Define the LSTM model.
num_components = 50
output_dim = y_train_scaled.shape[1]
params_size = tfp.layers.MixtureNormal.params_size(num_components, output_dim)  # size of the parameters

# Keep dropout on after model is trained
model = Sequential([
    LSTM(200, activation='tanh', input_shape=(X_train_scaled.shape[1], 1), return_sequences=False),
    Dropout(0.3),
    Dense(params_size),  # Adjust the number of neurons to match the dimensions of y
    tfp.layers.MixtureNormal(num_components, output_dim)
])


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
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.002), loss=nll)

# Define an early stopping callback
early_stopping = tfk.callbacks.EarlyStopping(
    monitor="val_loss",  # monitor the validation loss
    patience=10,  # if the validation loss does not improve for 10 epochs, stop training
    mode='min'
)

# Fit the model
history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=200,
    batch_size=32,
    validation_data=(x_val_scaled, y_val_scaled),
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
# %%
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

def calculate_crps(y_true, y_preds):
    n_obs, n_features = y_true.shape
    crps_values = np.zeros(n_features)

    for i in range(n_features):
        for j in range(n_obs):
            crps_values[i] += ps.crps_ensemble(y_true[j, i], y_preds[:, j, i])
        crps_values[i] /= n_obs

    crps = np.mean(crps_values)
    return crps

sample_num = 200
predicted_distributions = model(x_test_scaled)
preds = predicted_distributions.sample(sample_num)
preds = preds.numpy()

transformed_preds = np.zeros((sample_num,200,270))

for i in range(preds.shape[0]):
    original_scale_preds = scaler_y.inverse_transform(preds[i])
    transformed_preds[i] = original_scale_preds

mean_preds = np.mean(transformed_preds, axis=0)
conf_interval_lower = np.percentile(transformed_preds, 2.5, axis=0)
conf_interval_upper = np.percentile(transformed_preds, 97.5, axis=0)

rmse = calculate_rmse(y_test, mean_preds)
rmspe = calculate_rmspe(y_test, mean_preds)
crps = calculate_crps(y_test, transformed_preds)


print(f"CRPS: {crps}")
print(f"RMSE: {rmse}, RMSPE: {rmspe}")
#%%

rng = default_rng()
sample_index = rng.integers(0, x_test_scaled.shape[0])


for i in range(1600):
    plt.plot(y_train[i,:],color='grey',alpha=0.15)
for i in range(100):
    plt.plot(transformed_preds[i,sample_index,:],color='blue',alpha=0.15)
plt.plot(y_test[sample_index], label='Test',color='red')
plt.legend()
plt.show()


# CRPS: 1.084699502213776
# RMSE: 2.1607455920713665, RMSPE: 0.08831678029901606