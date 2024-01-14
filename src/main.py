import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import mnist

# %%
tf.keras.backend.clear_session()
print("TensorFlow version:", tf.__version__)

# %% load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# %% create basic model
model = models.Sequential(
    [
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(16, activation="relu"),
    ]
)
model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
model.summary()

# %% fit model
history = model.fit(
    x_train,
    y_train,
    epochs=5,
)
