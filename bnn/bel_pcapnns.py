import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl
from tensorflow.keras.layers import InputLayer, Dense

import matplotlib.pyplot as plt
from numpy.random import default_rng

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras


data = pd.read_pickle('datags.pkl')
time_steps = 30
X = data[:, :time_steps]
y = data[:, time_steps:]
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

pca_X = PCA(n_components=5)
X_pca = pca_X.fit_transform(X_scaled)
pca_Y = PCA(n_components=5)
y_pca = pca_Y.fit_transform(y_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y_pca, test_size=0.2, random_state=42)

hidden_units = 100
output_dim = y_train.shape[1]

model = tf.keras.Sequential([
    InputLayer(input_shape=X_train.shape[1]),
    Dense(hidden_units, activation='relu'),
    Dense(tfp.layers.MultivariateNormalTriL.params_size(output_dim)),
    tfp.layers.MultivariateNormalTriL(event_size=output_dim),
])


model.compile(loss=lambda y, distribution: -distribution.log_prob(y),
              optimizer=tf.optimizers.Adam(learning_rate=0.01))

history = model.fit(X_train, y_train, epochs=200, verbose=False, validation_data=(X_test, y_test))


plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

sample_num = 100
predicted_distributions = model(X_test)
preds = predicted_distributions.sample(sample_num)

pca_ytest = pca_Y.inverse_transform(y_test)
real_ytest = scaler_y.inverse_transform(pca_ytest)
transformed_preds = np.zeros((sample_num,400,270))
pca_inverted = pca_Y.inverse_transform(preds)
for i in range(preds.shape[0]):
    original_scale_preds = scaler_y.inverse_transform(pca_inverted[i])
    transformed_preds[i] = original_scale_preds


mean_preds = np.mean(transformed_preds, axis=0)
conf_interval_lower = np.percentile(transformed_preds, 2.5, axis=0)
conf_interval_upper = np.percentile(transformed_preds, 97.5, axis=0)

# %%
rng = default_rng()
sample_index = rng.integers(0, X_test.shape[0])

data_to_plot = [preds[:, sample_index, i] for i in range(5)]

plt.title(f'Distribution of Predictions for Sample {sample_index+1} with Actual Values')

violin = plt.violinplot(y_train, positions=range(1, 6), showmeans=True, showmedians=True)
violin = plt.violinplot(data_to_plot, positions=range(1, 6), showmeans=True, showmedians=True)


for i in range(5):
    plt.scatter(i+1, y_test[sample_index, i], color='red', zorder=3, label='Actual Value' if i == 0 else "")

plt.legend()
plt.xticks(range(1,6), [f'D {i+1}' for i in range(5)])
plt.show()
for i in range(2000):
    plt.plot(y[i,:],color='grey',alpha=0.05)

for i in range(100):
    plt.plot(transformed_preds[i,sample_index,:],color='red',alpha=0.5)

plt.plot(real_ytest[sample_index], label='Test',color='green')
plt.plot(mean_preds[sample_index], label='Predicted Mean')
# plt.fill_between(range(mean_preds.shape[1]), conf_interval_lower[sample_index], conf_interval_upper[sample_index], color='red', alpha=0.2, label='95% Confidence Interval')

plt.title(f'Prediction with 95% Confidence Interval for Sample {sample_index}')
plt.legend()
plt.show()
