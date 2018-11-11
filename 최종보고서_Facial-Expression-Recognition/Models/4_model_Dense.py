from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np
# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
categories = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
nb_classes = len(categories)
image_size = 48
def main():
    global X_test,y_test
    X_train, X_test, y_train, y_test = np.load("./CNN.npy")
    # 데이터 정규화하기
    X_train = X_train.reshape(X_train.shape[0], image_size * image_size).astype('float32') / 256
    X_test = X_test.reshape(X_test.shape[0], image_size * image_size).astype('float32') / 256
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test  = np_utils.to_categorical(y_test, nb_classes)
    # 모델을 훈련하고 평가하기
    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)
# 모델 구축하기
def build_model(in_shape):
    model = Sequential()
    model.add(Dense(512, input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
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
    hdf5_file = "./Dense_model.hdf5"
    model.save_weights(hdf5_file)
    return model
# 모델 평가하기
def model_eval(model, X, y):
    score = model.evaluate(X, y)
    print('loss=', score[0])
    print('accuracy=', score[1])
if __name__ == '__main__':
    main()
