import tensorflow as tf 
import keras
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())


import numpy as np

import cv2
import os

x_test=[]
y_test=[]
x_train=[]
y_train=[]

i=0
for l1 in os.listdir(" /home/ubuntu/ / "):
    i=i+1
    for filename in os.listdir(" /home/ubuntu/ /l1 "):
        img = cv2.imread(os.path.join(os.path.expanduser('~'),'',l1, ,filename))
        x_test.append(img)
        y_test.append(i)

i=0
for l1 in os.listdir(" /home/ubuntu/ / "):
    i=i+1
    for filename in os.listdir(" /home/ubuntu/ /l1 "):
        img = cv2.imread(os.path.join(os.path.expanduser('~'),'',l1, ,filename))
        x_train.append(img)
        y_train.append(i)

x_test=np.array(x_test)
y_test=np.array(y_test)
x_train=np.array(x_train)
y_train=np.array(y_train)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

batch_size = 32
num_classes = 60
epochs = 12

# input image dimensions
img_rows, img_cols = 100, 100

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
input_shape = (img_rows, img_cols, 3)

model = Sequential()
model.add(Conv2D(128, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              verbose=1
              metrics=['accuracy'])
model.save(raku_fruit.h5)