import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cross_validation import train_test_split

file_train = 'C:/Users/Chris/Desktop/train.csv' #loads training data
df = pd.read_csv(file_train);

file_pred = "C:/Users/Chris/Desktop/test.csv"; #loads testing data
pf = pd.read_csv(file_pred);

Y = df.loc[:, df.columns == 'label']; #features
X = df.loc[:, df.columns != 'label']; #target

Y = np.array(Y,dtype='float32'); #turn to array of floats for tensorflow
X[X!=0]=1 #changing all non-zero values to 1 so it will be a binary matrix
X = np.array(X,dtype='float32'); #turn to array of floats for tensorflow

pf[pf!=0]=1  #changing all non-zero values to 1 so it will be a binary matrix
X_pred = np.array(pf,dtype='float32'); #turn to array of floats for tensorflow


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25) #hold 25% of data as test and use 75% to train

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
]);

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']);

model.fit(X_train, Y_train, epochs=30); #train the model / result is 99% cross validation at epochs = 300

Y_pred = model.predict_classes(X_pred, verbose = 1) #apply model to new dataset

def write_preds(Y_pred, fname):
    pd.DataFrame({"ImageId": list(range(1,len(Y_pred)+1)), "Label": Y_pred}).to_csv(fname, index=False, header=True)

write_preds(Y_pred, "mnist_pred.csv")

check = 'C:/Users/Chris/mnist_pred.csv'
cf = pd.read_csv(check);
print(cf)