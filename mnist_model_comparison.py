# Import all necessary dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical

# Load the CSV file 
df = pd.read_csv('mnist.csv')  
print(df.head())
print(df.shape)

# Separate features (X) and labels (y)
X = df.drop("label", axis=1)
y = df["label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize pixel values to be between 0 and 1
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Reshape data for the fully connected models (2D images)
X_train_img = X_train.values.reshape(-1, 28, 28)
X_test_img = X_test.values.reshape(-1, 28, 28)

# One-hot encode the labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Define and train the Perceptron model
model_per = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(10, activation="softmax")
])
model_per.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
history_per = model_per.fit(X_train_img, y_train_cat, epochs=6, batch_size=32, validation_data=(X_test_img, y_test_cat), verbose=1)
loss, acc_per = model_per.evaluate(X_test_img, y_test_cat, verbose=1)
print("Perceptron Accuracy:", acc_per)

# Define and train the ANN model
model_ann = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(10, activation="softmax")
])
model_ann.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history_ann = model_ann.fit(X_train_img, y_train_cat, epochs=6, batch_size=32, validation_data=(X_test_img, y_test_cat), verbose=1)
loss, acc_ann = model_ann.evaluate(X_test_img, y_test_cat, verbose=1)
print("ANN Accuracy:", acc_ann)

# Prepare data for the CNN model (4D input)
X_train_cnn = X_train.values.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.values.reshape(-1, 28, 28, 1)

# Define and train the CNN model
model_cnn = Sequential([
    Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")
])
model_cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history_cnn = model_cnn.fit(X_train_cnn, y_train_cat, epochs=6, batch_size=32, validation_data=(X_test_cnn, y_test_cat), verbose=1)
loss, acc_cnn = model_cnn.evaluate(X_test_cnn, y_test_cat, verbose=1)
print("CNN Accuracy:", acc_cnn)

# Compare the accuracies of the models
models = ["Perceptron", "ANN", "CNN"]
accuracies = [acc_per, acc_ann, acc_cnn]

plt.figure(figsize=(6,4))
plt.bar(models, accuracies)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.ylim(0, 1)
plt.show()