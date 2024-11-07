# 1.Importing Libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# 2.Generate some toy data
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, size=(100,))

# 3.Define the neural network architecture
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 4.Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5.Train the model
model.fit(X, y, epochs=10, validation_split=0.2)

# 6.Evaluate the model
test_loss, test_acc = model.evaluate(X, y)
print(f"Test accuracy: {test_acc:.4f}")