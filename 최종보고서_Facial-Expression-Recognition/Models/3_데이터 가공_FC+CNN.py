import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import pandas as pd

csv = pd.read_csv('../fer2013.csv')

# limit=100
# csv=csv[0:limit]

csv_data = [[np.uint8(e) for e in i.split(" ")] for i in csv["pixels"]]
csv_label = [int(i) for i in csv["emotion"]]

X=[]
for index,data in enumerate(csv_data):
	data=np.asarray(data)
	data=np.reshape(data,(48,48,1))
	X.append(data)
	print(index,"완료")

X=np.array(X)

train_data, test_data, train_label, test_label = \
train_test_split(X, csv_label)

xy = (train_data, test_data, train_label, test_label)
np.save("./CNN.npy", xy)
print("ok!")