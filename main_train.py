import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical

image_file='dataset/'

no_tumor= os.listdir(image_file+ 'no/')
glioma_tumor= os.listdir(image_file+ 'glioma/')
meningioma_tumor= os.listdir(image_file+ 'meningioma/')
pituitary_tumor= os.listdir(image_file+ 'pituitary/')
datasets=[]
label=[]
INPUT_SIZE = 64

for i, image_name in enumerate(glioma_tumor):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_file+'glioma/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        datasets.append(np.array(image))
        label.append(0)
        
        
for i, image_name in enumerate(meningioma_tumor):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_file+'meningioma/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        datasets.append(np.array(image))
        label.append(1)
        
for i, image_name in enumerate(no_tumor):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_file+'no/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        datasets.append(np.array(image))
        label.append(2)
        
for i, image_name in enumerate(pituitary_tumor):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_file+'pituitary/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        datasets.append(np.array(image))
        label.append(3)
        
        
datasets=np.array(datasets)
label=np.array(label)


x_train, x_test, y_train, y_test= train_test_split(datasets, label, test_size=0.1, train_size=0.9, random_state=0)

x_train=normalize(x_train, axis=1)       
x_test=normalize(x_test, axis=1)  

# y_train=to_categorical(y_train, num_classes=(2))
# y_test=to_categorical(y_test, num_classes=(2))


#Model building

model=Sequential()

model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=(16), verbose=1, epochs=1, validation_data=(x_test, y_test), shuffle=(False))

model.save('model_final10EpochsCategorical.h5')



        
