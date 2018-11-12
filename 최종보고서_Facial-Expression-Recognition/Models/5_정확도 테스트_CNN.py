from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import sys
import numpy as np

def build_model(in_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), 
        padding='same',
        input_shape=in_shape,
        activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])
    return model

def model_eval(model, X, y):
    score = model.evaluate(X, y)
    print('loss=', score[0])
    print('accuracy=', score[1])


categories = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
nb_classes = len(categories)
image_size = 48
X_train, X_test, y_train, y_test = np.load("./CNN.npy")

# 데이터 정규화하기
X_train = X_train.astype("float") / 256
X_test  = X_test.astype("float")  / 256
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test  = np_utils.to_categorical(y_test, nb_classes)

model = build_model(X_train.shape[1:])
model.load_weights('CNN_model.hdf5')
# 데이터 예측하기 --- (※4)
# html = ""
# pre = model.predict(X)
model_eval(model,X_test,y_test)