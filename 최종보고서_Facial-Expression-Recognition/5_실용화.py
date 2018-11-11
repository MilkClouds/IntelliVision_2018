from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import sys, os, cv2, dlib, glob
from PIL import Image
import numpy as np

categories = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
nb_classes = len(categories)

def build_model(in_shape):
    model = Sequential()
    model.add(Conv2D(32, 3, 3, 
        border_mode='same',
        input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, 3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) 
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])
    return model

modelName='./Method_CNN/CNN_model.hdf5'
model = build_model((48,48,1))
model.load_weights(modelName)


def getExpression(imgs,model,returnMode='rate'):
	X = []
	files = []
	for image in imgs:
		img = Image.open(image)
		img = img.convert("L")
		img = img.resize((48,48))
		in_data = np.asarray(img)
		in_data = np.reshape(in_data,(48,48,1))
		X.append(in_data)
		files.append(image)
	X = np.array(X)
	X = X.astype("float") / 256
	pre = model.predict(X)

	if returnMode=='rate':
		returnRate=[] #
		for index,data in enumerate(pre):
			d={}
			for a,b in enumerate(categories):
				d[b]=(data[a]*np.float32(100)).round(3)
			returnRate.append(d)
		return returnRate

	returnList={}
	for index,data in enumerate(pre):
		y=data.argmax()
		returnList[os.path.basename(imgs[index])]=categories[y]
	return returnList

def getExpression1(img,model):
	img = Image.open(image)
	img = img.convert("L")
	img = img.resize((48,48))
	in_data = np.asarray(img)
	in_data = np.reshape(in_data,(48,48,1))
	X = np.array([in_data])
	X = X.astype("float") / 256
	pre = model.predict(X)
	return pre

imgs=glob.glob('./images/Surprise/*.png')
if len(imgs)<1: raise(Exception("no input"))
exp=getExpression(imgs,model,'1')

print(exp)