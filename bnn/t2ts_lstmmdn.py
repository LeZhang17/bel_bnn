import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from numpy.random import default_rng

tfd = tfp.distributions

# %% 1st step: First 10 years T to the following 3 years T

data = pd.read_pickle('datagt.pkl')

time_steps = 100

X = data[:, :time_steps]
y = data[:,time_steps:time_steps+30]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

num_components = 5
output_size = y_test.shape[1]

model = Sequential([
    LSTM(64, input_shape=(time_steps, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(tfp.layers.MixtureNormal.params_size(num_components, output_size)),
    tfp.layers.MixtureNormal(num_components, output_size)
])


model.compile(loss=lambda y, model: -model.log_prob(y),
              optimizer=Adam(learning_rate=0.001))

y_train_scaled = y_train_scaled.reshape(-1, output_size)
y_test_scaled = y_test_scaled.reshape(-1, output_size)

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=100, batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
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
sample_num = 10000
predicted_distributions = model(X_test_scaled)
preds = predicted_distributions.sample(sample_num)
preds = preds.numpy()

transformed_preds = np.zeros((sample_num,preds.shape[1],preds.shape[2]))

for i in range(preds.shape[0]):
    original_scale_preds = scaler_y.inverse_transform(preds[i])
    transformed_preds[i] = original_scale_preds

mean_preds = np.mean(transformed_preds, axis=0)
conf_interval_lower = np.percentile(transformed_preds, 5, axis=0)
conf_interval_upper = np.percentile(transformed_preds, 95, axis=0)
conf_interval_c = np.percentile(transformed_preds, 50, axis=0)

# %% 2nd Step: 10 years T to predict 10-13 years S

# X2_test = np.concatenate((X_test, mean_preds), axis=1)
# X2_test = mean_preds
X2 = data[:,:time_steps]
y2 = pd.read_pickle('datags.pkl')[:,time_steps:time_steps+30]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

scaler_X2 = StandardScaler()
scaler_y2 = StandardScaler()

X2_train_scaled = scaler_X2.fit_transform(X2_train)
X2_test_scaled = scaler_X2.transform(X2_test)

y2_train_scaled = scaler_y2.fit_transform(y2_train)
y2_test_scaled = scaler_y2.transform(y2_test)

# %%
num_components2 = 5
output_size2 = y2_test.shape[1]

model_2 = Sequential([
    LSTM(64, input_shape=(X2_test.shape[1], 1), return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(tfp.layers.MixtureNormal.params_size(num_components2, output_size2)),
    tfp.layers.MixtureNormal(num_components2, output_size2)
])

model_2.compile(loss=lambda y, model: -model.log_prob(y),
              optimizer=Adam(learning_rate=0.001))

early_stopping_2 = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

y2_train_scaled = y2_train_scaled.reshape(-1, output_size2)
y2_test_scaled = y2_test_scaled.reshape(-1, output_size2)

history_2 = model_2.fit(
    X2_train_scaled, y2_train_scaled,
    epochs=100, batch_size=32,
    validation_split=0.2, callbacks=[early_stopping_2], verbose=1)

plt.figure()
plt.plot(history_2.history['loss'], label='Training Loss')
plt.plot(history_2.history['val_loss'], label='Validation Loss')
plt.title("Model 2 Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%
predicted_distributions2 = model_2(X2_test_scaled)
preds2 = predicted_distributions2.sample(sample_num)
preds2 = preds2.numpy()

transformed_preds2 = np.zeros((sample_num,preds2.shape[1],preds2.shape[2]))

for i in range(preds2.shape[0]):
    original_scale_preds = scaler_y2.inverse_transform(preds2[i])
    transformed_preds2[i] = original_scale_preds

mean_preds2 = np.mean(transformed_preds2, axis=0)
conf_interval_lower2 = np.percentile(transformed_preds2, 5, axis=0)
conf_interval_upper2 = np.percentile(transformed_preds2, 95, axis=0)
conf_interval_c2 = np.percentile(transformed_preds2, 50, axis=0)
# %% visualisation

rng = default_rng()
sample_index = rng.integers(0, X_test_scaled.shape[0])
x = np.arange(0,30,1)

plt.plot(y_train.transpose(),color='gray',alpha=0.02)
plt.fill_between(x,conf_interval_lower[sample_index],conf_interval_upper[sample_index],
                 color='blue', alpha=0.5, label='Confidence Interval')
# plt.plot(conf_interval_lower[sample_index])
# plt.plot(conf_interval_upper[sample_index])
plt.plot(mean_preds[sample_index], label='pre')
plt.plot(y_test[sample_index], label='Test')
plt.legend()
plt.show()


plt.plot(y2_train.transpose(),color='gray',alpha=0.02)
plt.fill_between(x, conf_interval_lower2[sample_index], conf_interval_upper2[sample_index],
                 color='blue', alpha=0.5, label='90% Confidence Interval')
# plt.plot(conf_interval_lower2[sample_index])
# plt.plot(conf_interval_upper2[sample_index])
plt.plot(mean_preds2[sample_index], label='Pre_mean')
plt.plot(y2_test[sample_index], label='Test')
plt.legend()
plt.show()

