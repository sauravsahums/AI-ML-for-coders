import tensorflow as tf

data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

# Reshape each array to have that extra dimension. 28 * 28 is the num of pixels.
# 1 is the num of color channels. 1 for Grayscale img or 3 for Color img.
training_images = training_images.reshape(60000, 28, 28, 1)  # Color Images RGB
training_images = training_images / 255.0

test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',   # 64 filters, 3*3 kernel
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),  # Down-sampling feature map using 2X2 pooling window
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # input_shape not passed, auto-inferred.
    tf.keras.layers.MaxPooling2D(2, 2),  # Further reduce spatial dimensions
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])

# Adding convolutions to the NN increases its ability to classify images. Accuracy improvement 89% -> 99%

model.summary()
# >> Outputs <<
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step
# [1.8315457e-29 1.5342983e-24 1.0931570e-29 6.8966348e-31 5.4015800e-37
#  6.2606978e-26 5.8647452e-30 1.3640908e-18 5.9792108e-29 1.0000000e+00]
# 9
# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
# │ conv2d (Conv2D)                 │ (None, 26, 26, 64)     │           640 │  ## On running 3 x 3 filter, 1-pixel border is lost. First possible filter starts at {1,1}
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ max_pooling2d (MaxPooling2D)    │ (None, 13, 13, 64)     │             0 │  ## Pooling layer 2 X 2 makes the size of img 'half' on each axis.
# ├─────────────────────────────────┼────────────────────────┼───────────────┤  ## Maxpooling layer doesn't learn anything, they just reduce the img, so, there are no learned parameters there.
# │ conv2d_1 (Conv2D)               │ (None, 11, 11, 64)     │        36,928 │  ## Multiplied across prev 64 filters, each with 9 parameters. (64*(64*9)) = 36,928 parameters the network needs to learn.
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ max_pooling2d_1 (MaxPooling2D)  │ (None, 5, 5, 64)       │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ flatten (Flatten)               │ (None, 1600)           │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dense (Dense)                   │ (None, 128)            │       204,928 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dense_1 (Dense)                 │ (None, 10)             │         1,290 │
# └─────────────────────────────────┴────────────────────────┴───────────────┘
#  Total params: 731,360 (2.79 MB)   ## Number of 5 X 5 result images.
#  Trainable params: 243,786 (952.29 KB)
#  Non-trainable params: 0 (0.00 B)
#  Optimizer params: 487,574 (1.86 MB)