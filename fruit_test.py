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
i=-1
for l1 in os.listdir("/home/ubuntu/fruit_val/Validation"):
    i=i+1
    path='/home/ubuntu/fruit_val/Validation/%s'%l1
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(os.path.expanduser('~'),'fruit_val','Validation',l1,filename))
        x_test.append(img)
        y_test.append(i)
x_test=np.array(x_test)
y_test=np.array(y_test)
x_test /= 255
from keras.models import load_model

model=load_model("raku_fruit.h5")
import keras

y_test1 = keras.utils.to_categorical(y_test, 60)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])