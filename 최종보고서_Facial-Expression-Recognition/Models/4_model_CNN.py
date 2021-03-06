from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np
categories = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
nb_classes = len(categories)
def main():
    global X_test,y_test
    X_train, X_test, y_train, y_test = np.load("./CNN.npy")
    # 데이터 정규화하기
    X_train = X_train.astype("float") / 256
    X_test  = X_test.astype("float")  / 256
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test  = np_utils.to_categorical(y_test, nb_classes)
    # 모델을 훈련하고 평가하기
    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)
# 모델 구축하기
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
# 모델 훈련하기
def model_train(X, y):
    model = build_model(X.shape[1:])
    # model.summary()
    model.fit(X, y, batch_size=32, epochs=30,validation_data=(X_test, y_test))
    # 모델 저장하기
    hdf5_file = "./CNN_model.hdf5"
    model.save_weights(hdf5_file)
    return model
# 모델 평가하기
def model_eval(model, X, y):
    score = model.evaluate(X, y)
    print('loss=', score[0])
    print('accuracy=', score[1])
if __name__ == "__main__":
    main()