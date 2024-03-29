import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from kerastuner import HyperModel
from kerastuner.tuners import Hyperband

# 加载数据
data = pd.read_pickle('datags.pkl')

# 数据预处理
time_steps = 30
X = data[:, :time_steps]
y = data[:, time_steps:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_test, x_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
scaler_X.fit(X_train)
scaler_y.fit(y_train)

X_train_scaled = scaler_X.transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
x_val_scaled = scaler_X.transform(x_val)
x_test_scaled = scaler_X.transform(x_test)
y_train_scaled = scaler_y.transform(y_train)
y_test_scaled = scaler_y.transform(y_test)
y_val_scaled = scaler_y.transform(y_val)

pca_X = PCA(n_components=.99)
pca_Y = PCA(n_components=.99)
pca_X.fit(X_train_scaled)
pca_Y.fit(y_train_scaled)

X_train_pca = pca_X.transform(X_train_scaled)
X_test_pca = pca_X.transform(X_test_scaled)
x_val_pca = pca_X.transform(x_val_scaled)
x_test_pca = pca_X.transform(x_test_scaled)
y_train_pca = pca_Y.transform(y_train_scaled)
y_test_pca = pca_Y.transform(y_test_scaled)
y_val_pca = pca_Y.transform(y_val_scaled)

X_train_pca = X_train_pca.reshape((X_train_pca.shape[0], X_train_pca.shape[1], 1))
X_test_pca = X_test_pca.reshape((X_test_pca.shape[0], X_test_pca.shape[1], 1))


# 定义LSTM模型和超参数搜索
class LSTMHyperModel(HyperModel):
    def __init__(self, input_shape, output_dim):
        self.input_shape = input_shape
        self.output_dim = output_dim

    def build(self, hp):
        model = Sequential([
            LSTM(units=hp.Int('units1', min_value=50, max_value=200, step=50),
                 activation='tanh', input_shape=self.input_shape, return_sequences=False),
            Dropout(rate=hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)),
        ])

        num_components = hp.Int('num_components', min_value=10, max_value=100, step=10)
        params_size = tfp.layers.MixtureNormal.params_size(num_components, self.output_dim)

        model.add(Dense(params_size))
        model.add(tfp.layers.MixtureNormal(num_components, self.output_dim))

        model.compile(optimizer=tf.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                      loss=nll)

        return model


def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)


tuner = Hyperband(
    LSTMHyperModel(input_shape=X_train_pca.shape[1:], output_dim=y_train_pca.shape[1]),
    objective='val_loss',
    max_epochs=10,
    directory='kt_dir',
    project_name='lstm_optimization_with_num_components'
)

# 开始超参数搜索
tuner.search(X_train_pca, y_train_pca, epochs=200, validation_data=(x_val_pca, y_val_pca),
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')])

# 获取最佳模型和超参数
best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best hyperparameters found are:")
print(f"First LSTM layer units: {best_hps.get('units1')}")
print(f"Drop rate: {best_hps.get('dropout1')}")
print(f"Number of components: {best_hps.get('num_components')}")
print(f"Learning rate for the optimizer: {best_hps.get('learning_rate')}")

# 评估最佳模型
loss = best_model.evaluate(x_test_pca, y_test_pca)
print(f"Loss of the best model on test data: {loss}")

# Best val_loss So Far: 6.544004917144775
# Total elapsed time: 00h 01m 37s
# Best hyperparameters found are:
# First LSTM layer units: 150
# Drop rate: 0.1
# Number of components: 80
# Learning rate for the optimizer: 0.005862125721808125
# 7/7 [==============================] - 0s 2ms/step - loss: 6.5731
# Loss of the best model on test data: 6.573062896728516