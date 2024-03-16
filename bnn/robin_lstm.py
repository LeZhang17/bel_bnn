import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

cwd = os.getcwd()
data = pd.read_pickle(os.path.join(cwd, "bnn", 'datags.pkl'))

# Split data into predictors (X) and target (y)
time_steps = 30  # Past time steps you're using to predict future
X = data[:, :time_steps]
y = data[:, time_steps:]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Reshape for LSTM input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Reshape the data for LSTM layer
# LSTM expects input in the form of [samples, time steps, features]
# We must adjust the shape of the parameters based on your data's structure

# Note: This step assumes you have 1 feature per time step. Adjust if your data is different.
n_features = 1
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

# Define the LSTM model.
# Keep dropout on after model is trained
model = Sequential([
    LSTM(50, activation='relu', input_shape=(time_steps, n_features), return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation='relu', return_sequences=False),
    Dropout(0.2),
    Dense(y_train.shape[1])  # Adjust the number of neurons to match the dimensions of y
])

# Compile the model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# Fit the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
history = model.fit(X_train, y_train, epochs=1000, validation_split=0.1, callbacks=[early_stopping], batch_size=32)

# Plot training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Select test 161 and plot the predictions
test_sample = 161
x = X_test[test_sample]
y_true = y_test[test_sample]

# Reshape the sample for the model
x = x.reshape((1, time_steps, n_features))

# Predict
n_samples = 1000
# y_pred = model.predict(x)
y_pred = np.array([model(x, training=True) for _ in range(n_samples)])

# Inverse transform the predictions
y_pred = scaler_y.inverse_transform(y_pred.reshape(n_samples, -1))
y_true = scaler_y.inverse_transform(y_true.reshape(1, -1))

# Plot the predictions
# Plot the predictions
plt.figure(figsize=(10, 6))
for i in range(n_samples):
    plt.plot(y_pred[i], color="black", alpha=0.1)
plt.plot(y_true[0], label="True", color="red", linewidth=4)
plt.title("Samples and True Value")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.title(f"Test Sample {test_sample} Predictions with LSTM + Dropout")
plt.savefig("lstm_dropout.png", dpi=300, bbox_inches='tight')
plt.show()