import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl
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

# Issue here - you should fit the PCA to the training data and then only transform the test set
pca_X = PCA(n_components=5)
X_pca = pca_X.fit_transform(X_scaled)
pca_Y = PCA(n_components=5)
y_pca = pca_Y.fit_transform(y_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y_pca, test_size=0.2, random_state=42)

def posterior(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                       scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])


def prior(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])


hidden_units = 32

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=X_train.shape[1]),
    tfpl.DenseVariational(units=hidden_units,
                          make_prior_fn=prior,
                          make_posterior_fn=posterior,
                          kl_weight=1 / X_train.shape[0],
                          activation='relu'),


    tfpl.DenseVariational(units=y_train.shape[1],
                          make_prior_fn=prior,
                          make_posterior_fn=posterior,
                          kl_weight=1 / X_train.shape[0]),
])

model.compile(loss='mean_squared_error', optimizer=tf.optimizers.Adam(learning_rate=0.01))

history = model.fit(X_train, y_train, epochs=200, verbose=False,validation_data=(X_test, y_test))


plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

sample_num = 100
preds = [model(X_test) for _ in range(sample_num)]
preds = np.array([pred.numpy() for pred in preds])

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
