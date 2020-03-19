#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  16 09:59:53 2020

@author: dev
"""

#%%
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = '/home/hp/Downloads/grape-disease/Grape/train'
val_dir = '/home/hp/Downloads/grape-disease/Grape/test'

num_train=4062
num_val=1782
batch_size=32
num_epochs=200

train_datagen=ImageDataGenerator(rescale=1./255)
val_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode=('grayscale'),
        class_mode=('categorical')
       
        )

validation_generator=val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=32,
        color_mode=('grayscale'),
        class_mode=('categorical')
        
        )


#%%

model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(4,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001,decay=1e-6),metrics=['accuracy'])

model_info = model.fit_generator(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epochs,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size)

model.save('/home/hp/Downloads/grape-disease/grapes.h5')

#%%
from keras.models import load_model
import tensorflow as tf 
model = tf.keras.models.load_model('/home/hp/Downloads/grape-disease/grapes.h5')

import cv2
import numpy as np
import matplotlib.pyplot as plt

img= cv2.imread("/home/hp/Downloads/grape-disease/grape/Grape/Grape___Esca_(Black_Measles)/5d6ab00e-8d19-464e-b8fa-720307aebff0___FAM_B.Msls 0855.JPG")
#plt.imshow(img)
Labels=['black rot','Esca(black Measles)','Leaf_Blight(isariopsis_leaf_spot)','Healthy']
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img=cv2.resize(img,(48,48))
plt.imshow(img)
img=img.reshape(48,48,1)

img=np.expand_dims(img,axis=0)
print(img.shape)
ret=model.predict(img)
print(np.argmax(ret))
print(Labels[np.argmax(ret)])

#%%
import tensorflow as tf 
model = tf.keras.models.load_model('/home/hp/Downloads/grape-disease/grapes.h5')

   
   
