import os
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from kerastuner import HyperModel
from kerastuner.tuners import Hyperband


# 定义负对数似然损失函数
def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)


# 超参数搜索模型定义
class HyperModelBuilder(HyperModel):
    def __init__(self, input_shape, output_dim):
        self.input_shape = input_shape
        self.output_dim = output_dim

    def build(self, hp):
        inputs = Input(shape=self.input_shape, name="input")

        # 允许Keras Tuner搜索隐藏层单元数
        hidden_units = hp.Int('hidden_units', min_value=10, max_value=200, step=50)
        x = Dense(hidden_units, activation="relu")(inputs)

        # 允许Keras Tuner搜索num_components
        num_components = hp.Int('num_components', min_value=2, max_value=100, step=40)
        params_size = tfp.layers.MixtureNormal.params_size(num_components, self.output_dim)
        x = Dense(params_size, activation=None, name="output")(x)

        outputs = tfp.layers.MixtureNormal(num_components, self.output_dim)(x)

        model = Model(inputs=inputs, outputs=outputs, name="model")

        # 允许Keras Tuner搜索学习率
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=nll)

        return model


# 加载和准备数据
data = pd.read_pickle('datags.pkl')  # 调整为你的数据路径
time_steps = 30
X = data[:, :time_steps]
y = data[:, time_steps:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_X = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train)
X_train_scaled = scaler_X.transform(X_train)
y_train_scaled = scaler_y.transform(y_train)
pca_X = PCA(n_components=.99).fit(X_train_scaled)
pca_Y = PCA(n_components=.99).fit(y_train_scaled)
X_train_pca = pca_X.transform(X_train_scaled)
y_train_pca = pca_Y.transform(y_train_scaled)

# 设置Keras Tuner
tuner = Hyperband(
    HyperModelBuilder(input_shape=X_train_pca.shape[1], output_dim=y_train_pca.shape[1]),
    objective='val_loss',
    max_epochs=10,
    directory='kt_hyperband',
    project_name='mixture_density_network'
)

# 运行超参数搜索
tuner.search(X_train_pca, y_train_pca, validation_split=0.2, epochs=50,
             callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=5)])

# 获取并打印最佳超参数
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The best number of hidden units in the first dense layer: {best_hps.get('hidden_units')}
The best number of components: {best_hps.get('num_components')}
The best learning rate for the optimizer: {best_hps.get('learning_rate')}
""")

# The best number of hidden units in the first dense layer: 110
# The best number of components: 42
# The best learning rate for the optimizer: 0.006381467849613307