import tensorflow as tf
from tensorflow.python.data.experimental.ops.distribute import batch_sizes_for_worker

print(tf.__version__)

data = tf.keras.datasets.fashion_mnist

# 60k tr - 10k te images. Labels are 0-9 value.
(training_images, training_labels), (test_images, test_labels) = data.load_data()

# All pixels in images are grayscale with [0-255] value.
# Divide to get decimal value in range [0, 1]. This process is called normalizing the image.
training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),      # rectified linear unit
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',   # loss function
              metrics=['accuracy'])

# Train the network by fitting the training images to the training labels over 5 epochs.
# Note there are 60,000 ÷ 32 ≈ 1875 batches. Default batch_size in Keras is 32.
# Expected Output:
# Epoch 5/5
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 639us/step - accuracy: 0.8905 - loss: 0.2946
model.fit(training_images, training_labels, epochs=50)

# Evaluate the model
# Note there are 10,000 ÷ 32 ≈ 313 batches
model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0]) # prints values of 10 output neurons
print(test_labels[0]) # the label is 9. You might notice that the classification[0][9] is largest by far.
# Consider its value came out as 0.915, it would mean that there's 91.5% chance that the item of clothing at index 0 is label 9.

# Experience "overfitting" by increasing the epochs to higher values (like 50). The model become overspecialized to the trData.