# TensorFlow Neural Network Demos

A collection of neural network experiments using TensorFlow and Keras — ranging from basic linear regression to image classification using CNNs on the Fashion MNIST dataset.

---

## 📌 Project Overview

This project explores:

- ✅ Simple linear regression using a single neuron
- 👕 Fashion MNIST classification with dense layers
- ⏹️ Early stopping with custom callbacks
- 🧩 CNN architecture for high-accuracy image classification

Each section demonstrates key machine learning concepts using minimal and clean TensorFlow/Keras code.

---

## 🛠️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/sauravsahums/AI-ML-for-coders.git
cd AI-ML-for-coders
```

### 2. Install dependencies

```bash
pip install tensorflow numpy
```

---

## 🚀 Examples

### 🔹 1. Simple Linear Model

```python
model = Sequential([Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(xs, ys, epochs=500)
print(model.predict([10.0]))  # Output should be close to 19
```

### 🔹 2. Dense Neural Network for Fashion MNIST

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=50)
```

### 🔹 3. Early Stopping with Custom Callback

```python
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            self.model.stop_training = True
```

### 🔹 4. CNN for Fashion MNIST

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.fit(training_images, training_labels, epochs=50)
```

---

## Key Concepts

- **Normalization**: Crucial for training stability and performance.
- **Activation Functions**: `ReLU` for hidden layers, `Softmax` for multiclass classification.
- **Overfitting**: Controlled via callbacks or validation monitoring.
- **CNNs**: Effective for image recognition by learning spatial hierarchies.

---

## 📈 Sample Output

```text
Reached 95% accuracy so cancelling training
Model: "sequential"
Total params: 731,360
Trainable params: 243,786
Test accuracy: ~99%
```

---

## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)

---

## 📄 License

This project is licensed under the MIT License.
