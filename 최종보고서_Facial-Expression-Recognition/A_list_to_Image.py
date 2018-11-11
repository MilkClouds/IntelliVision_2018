import cv2,sys
import pandas as pd
import numpy as np
from PIL import Image
import os

rootDir='./images/'

csv = pd.read_csv('fer2013.csv')
csv=csv[0:1]
# csv=csv[28000:35000]
csv_data = [[np.uint8(e) for e in i.split(" ")] for i in csv["pixels"]]
csv_label = [int(i)for i in csv["emotion"]]

categories = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

for i in categories:
	if not os.path.exists(rootDir+i): os.mkdir(rootDir+i)

for index,data in enumerate(csv_data):
	print(index)
	# if index<10000:continue
	r=np.reshape(data,(48,48))
	cv2.imwrite(rootDir+categories[csv_label[index]]+"/"+str(index+0)+".png",r)
	# cv2.imshow(categories[int(csv_label[index])],r)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()