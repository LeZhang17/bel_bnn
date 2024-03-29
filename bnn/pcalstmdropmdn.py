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

# Initialize PCA - keep 99% of the variance
pca_X = PCA(n_components=.99)
pca_Y = PCA(n_components=.99)

# Fit the PCA
pca_X.fit(X_train_scaled)
pca_Y.fit(y_train_scaled)

# Transform the data
X_train_pca = pca_X.transform(X_train_scaled)
X_test_pca = pca_X.transform(X_test_scaled)
x_val_pca = pca_X.transform(x_val_scaled)
x_test_pca = pca_X.transform(x_test_scaled)

y_train_pca = pca_Y.transform(y_train_scaled)
y_test_pca = pca_Y.transform(y_test_scaled)
y_val_pca = pca_Y.transform(y_val_scaled)

# Reshape for LSTM input
X_train_pca = X_train_pca.reshape((X_train_pca.shape[0], X_train_pca.shape[1], 1))
X_test_pca = X_test_pca.reshape((X_test_pca.shape[0], X_test_pca.shape[1], 1))

# Define the LSTM model.
num_components = 80
output_dim = y_train_pca.shape[1]
params_size = tfp.layers.MixtureNormal.params_size(num_components, output_dim)  # size of the parameters

# Keep dropout on after model is trained
model = Sequential([
    LSTM(150, activation='tanh', input_shape=(X_train_pca.shape[1], 1), return_sequences=False),
    Dropout(0.1),
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
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.004), loss=nll)

# Define an early stopping callback
early_stopping = tfk.callbacks.EarlyStopping(
    monitor="val_loss",  # monitor the validation loss
    patience=10,  # if the validation loss does not improve for 10 epochs, stop training
    mode='min'
)

# Fit the model
history = model.fit(
    X_train_pca,
    y_train_pca,
    epochs=200,
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

#%%

sample_num = 200
predicted_distributions = model(x_test_pca)
preds = predicted_distributions.sample(sample_num)
preds = preds.numpy()

transformed_preds = np.zeros((sample_num,200,270))

pca_inverted = pca_Y.inverse_transform(preds)

for i in range(preds.shape[0]):
    original_scale_preds = scaler_y.inverse_transform(pca_inverted[i])
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
sample_index = rng.integers(0, x_test_pca.shape[0])

data_to_plot = [preds[:, sample_index, i] for i in range(3)]

plt.title(f'Distribution of Predictions for Sample {sample_index+1} with Actual Values')
violin = plt.violinplot(y_train_pca, positions=range(1, 4), showmeans=True, showmedians=True)
violin = plt.violinplot(data_to_plot, positions=range(1, 4), showmeans=True, showmedians=True)
for i in range(3):
    plt.scatter(i+1, y_test_pca[sample_index, i], color='red', zorder=3, label='Actual Value' if i == 0 else "")
plt.legend()
plt.xticks(range(1,4), [f'D {i+1}' for i in range(3)])
plt.show()


for i in range(1600):
    plt.plot(y_train[i,:],color='grey',alpha=0.15)
for i in range(100):
    plt.plot(transformed_preds[i,sample_index,:],color='yellow',alpha=0.2,label='Pre' if i == 0 else None)
plt.plot(y_test[sample_index], label='Test',color='red',lw=3)
plt.plot(mean_preds[sample_index], label='Pre Mean',color='blue',lw=3)
plt.legend()
plt.show()

g = sns.JointGrid(x=y_train_pca[:, 0], y=y_train_pca[:, 1], space=0)
g = g.plot_joint(plt.scatter, color="m")
g = g.plot_marginals(sns.kdeplot, color="m", fill=True)
g.ax_joint.scatter(preds[:, sample_index, 0], preds[:, sample_index, 1], color='red')
sns.kdeplot(x=preds[:, sample_index, 0], ax=g.ax_marg_x, color="r", fill=True)
sns.kdeplot(y=preds[:, sample_index, 1], ax=g.ax_marg_y, color="r", fill=True)
g.ax_joint.scatter(y_test_pca[sample_index, 0], y_test_pca[sample_index, 1], color='blue', s=30, edgecolor='black', zorder=10)
plt.show()

# CRPS: 0.6478738382343
# RMSE: 1.4140247218066067, RMSPE: 0.061209870330701