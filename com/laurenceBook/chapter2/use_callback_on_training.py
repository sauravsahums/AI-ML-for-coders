import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    # Check the logs of the epoch. If accuracy is gt 0.95, stop training.
    # TensorFlow looks specifically for a method named 'on_epoch_end'
    def on_epoch_end(self, epoch, logs: {}):
        if(logs.get('accuracy')>0.95):
            print("\nReached 95% accuracy so cancelling training")
            self.model.stop_training = True

callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50,
          callbacks=[callbacks])

# Output paste :~
# Epoch 34/50
# 1857/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 650us/step - accuracy: 0.9531 - loss: 0.1277
# Reached 95% accuracy so cancelling training



