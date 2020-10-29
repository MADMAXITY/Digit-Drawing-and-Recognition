###Note : Unzip the data in the data folder before runnning this code.
###       Github does not allow file upload more than 25mb.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow import keras
from sklearn.model_selection import train_test_split

data = pd.read_csv("Data/digits.csv")

X, Y = data.drop("label", axis=1), data["label"]
X = X / 255.0

X = np.array(X).reshape(X.shape[0], 28, 28)
Y = keras.utils.to_categorical(Y, num_classes=10)

w = 10
h = 10
fig = plt.figure(figsize=(15, 15))
columns = 5
rows = 4
for i in range(1, columns * rows + 1):
    img = np.random.randint(10, size=(h, w))
    fig.add_subplot(rows, columns, i)
    loc = random.randint(0, len(X) - 1)
    plt.imshow(X[loc], cmap="gray")
plt.show()

X = X.reshape(-1, 28, 28, 1)

Xtrain, xval, Ytrain, yval = train_test_split(X, Y, test_size=0.22)


##### Model #####
model = keras.Sequential()

model.add(
    keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        input_shape=(28, 28, 1),
    )
)
model.add(keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))

model.add(keras.layers.MaxPooling2D((2, 2), strides=(1, 1), padding="same"))

model.add(
    keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation="relu", padding="same"
    )
)
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))

model.add(keras.layers.MaxPooling2D((2, 2), padding="same"))

model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(10, activation="softmax"))


keras.utils.plot_model(model, to_file="model.png", show_shapes=True)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(
    Xtrain, Ytrain, epochs=50, batch_size=32, validation_data=(xval, yval)
)

model.save("Trained_model.h5")
