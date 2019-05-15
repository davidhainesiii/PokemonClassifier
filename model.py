import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split

import keras
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation
from keras.models import Sequential
from keras.utils import to_categorical

data = pd.read_csv('./df_final.csv')

mask = np.random.rand(len(data)) < 0.8
train = data[mask].reset_index(drop=True)
test = data[~mask].reset_index(drop=True)

gen = ImageDataGenerator(vertical_flip=True, validation_split=0.2)

traingen = gen.flow_from_dataframe(train,
                                   directory='./images/comp/',
                                   x_col='fileid',
                                   y_col='Type1',
                                   class_mode='categorical',
                                   target_size=(224,224),
                                   subset='training')

validgen = gen.flow_from_dataframe(train,
                                   directory='./images/comp/',
                                   x_col='fileid',
                                   y_col='Type1',
                                   class_mode='categorical',
                                   target_size=(224,224),
                                   subset='validation')

testdatagen = ImageDataGenerator(vertical_flip=False)
testgen = testdatagen.flow_from_dataframe(test, directory='./images/comp/', x_col='fileid', y_col='Type1', target_size=(224,224))

model = Sequential()
model.add(Conv2D(32, kernel_size=2, padding='same', activation='relu',
                 input_shape=(224, 224, 3)))
model.add(Conv2D(32, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=2, activation='relu'))
model.add(Conv2D(64, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(18, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy')

STEP_SIZE_TRAIN=traingen.n//traingen.batch_size
STEP_SIZE_VALID=validgen.n//validgen.batch_size
step_size_test=testgen.n//testgen.batch_size

model.fit_generator(generator=traingen,
                    epochs=10,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                   validation_data=validgen,
                   validation_steps=STEP_SIZE_VALID)


