import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from kerastuner import HyperModel
from kerastuner.tuners import Hyperband


# 负对数似然损失函数
def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)


# 超参数模型构建类
class LSTMHyperModel(HyperModel):
    def __init__(self, input_shape, output_dim):
        self.input_shape = input_shape
        self.output_dim = output_dim

    def build(self, hp):
        model = Sequential([
            LSTM(
                units=hp.Int('lstm_units', min_value=50, max_value=200, step=50),
                activation='tanh',
                input_shape=self.input_shape,
                return_sequences=False  # 注意：这里设置为False，因为我们不在LSTM后面紧接另一个LSTM层
            ),
            Dropout(rate=hp.Float('dropout_rate', min_value=0, max_value=0.5, step=0.1)),
            Dense(
                tfp.layers.MixtureNormal.params_size(
                    hp.Int('num_components', min_value=10, max_value=100, step=10),
                    self.output_dim
                )
            ),  # 调整神经元数量以匹配输出维度
            tfp.layers.MixtureNormal(
                hp.Int('num_components', min_value=10, max_value=100, step=10),
                self.output_dim
            )
        ])

        model.compile(
            optimizer=tf.optimizers.Adam(
                hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
            ),
            loss=nll
        )

        return model


# 加载数据
data = pd.read_pickle('datags.pkl')  # 调整为你的数据路径
time_steps = 30
X = data[:, :time_steps]
y = data[:, time_steps:]

# 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_X = StandardScaler().fit(X_train)
X_train_scaled = scaler_X.transform(X_train)
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))  # 重塑为LSTM输入需要的形状

# 定义和运行超参数搜索
tuner = Hyperband(
    LSTMHyperModel(input_shape=X_train_scaled.shape[1:], output_dim=y_train.shape[1]),
    objective='val_loss',
    max_epochs=10,
    directory='my_dir',
    project_name='lstm_mdn'
)

tuner.search(
    X_train_scaled, y_train, epochs=50, validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
)

# 获取最佳超参数
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
Best hyperparameters:
- LSTM units: {best_hps.get('lstm_units')}
- Dropout rate: {best_hps.get('dropout_rate')}
- Number of components: {best_hps.get('num_components')}
- Learning rate: {best_hps.get('learning_rate')}
""")

# Best hyperparameters:
# - LSTM units: 200
# - Dropout rate: 0.30000000000000004
# - Number of components: 50
# - Learning rate: 0.0024174314667387863