# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model

![Uploading o6.png…]()


## DESIGN STEPS
### STEP 1: 
Import the required packages
### STEP 2: 
Load the dataset
### STEP 3: 
Scale the dataset
### STEP 4:
Use the one-hot encoder
### STEP 5:
Create the model
### STEP 6:
Compile the model
### STEP 7:
Fit the model
### STEP 8: 
Make prediction with test data and with an external data

## PROGRAM
~~~
NAME : v.charan sai
REF  : 212221240061
~~~
~~~
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
plt.imshow(single_image)
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add (layers. Input (shape=(28,28,1)))
model.add (layers.Conv2D (filters=32, kernel_size=(3,3), activation='relu')) 
model.add (layers.MaxPool2D (pool_size=(2,2)))
model.add (layers. Flatten())
model.add (layers.Dense (32, activation='relu'))
model.add (layers.Dense (16, activation='relu'))
model.add (layers.Dense (8, activation='relu'))
model.add (layers.Dense (10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)

metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

img = image.load_img('image18.jpeg')
type(img)

img = image.load_img('image18.jpeg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)   
print(x_single_prediction)
~~~
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![266796366-7843d2db-c292-4018-93f1-ce6549a36929](https://github.com/charansai0/mnist-classification/assets/94296221/10bb88ee-07ff-4524-b51b-c9e4d8c26dd4)



### Classification Report
![266796393-0393e7af-d4d8-4c2c-a6b1-2c7e050235f5](https://github.com/charansai0/mnist-classification/assets/94296221/0c9830e9-1c6f-44d7-a0bd-84583cf8acb5)

### Confusion Matrix
![266796419-d89bec76-0ecc-4974-abcc-b63a86cc0271](https://github.com/charansai0/mnist-classification/assets/94296221/7db1d4e5-d7b0-4156-9f2b-e0f30ab4046c)
### New Sample Data Prediction
![image18](https://github.com/charansai0/mnist-classification/assets/94296221/67fc99ca-f102-464d-9983-3af880a165f4)


## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
