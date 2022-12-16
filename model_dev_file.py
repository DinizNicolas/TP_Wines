#Temporary file

import keras
from keras import models
from keras import layers
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import json

def regression_model():
    model = models.Sequential()
    model.add(layers.Dense(11, input_shape=(11,), kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(11, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(1, kernel_initializer='normal', activation='linear'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=[keras.metrics.MeanAbsolutePercentageError(),keras.metrics.MeanAbsoluteError()])

    return model

def classification_model():
    model = models.Sequential()
    model.add(layers.Dense(64, input_shape=(11,), kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(10, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def preprocess_for_reg(data_):
    X = data_[:,0:11]
    Y = data_[:,-2]
    Y = Y/10
    return (X,Y)

def preprocess_for_clas(data_):
    X = data_[:,0:11]
    Y = data_[:,-2]
    Y_ = []
    for y in Y:
        zeros = [0 for i in range(10)]
        zeros[int(y)] = 1
        Y_.append(zeros)
    return (X,np.array(Y_))
    

data = pd.read_csv('./data/Wines.csv')
for name in data.columns:
    data[name] = data[name].astype(float)

data_ = data.values

# X,Y = preprocess_for_reg(data_)
X,Y = preprocess_for_clas(data_)

min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)
infos_save = []
for i in range(len(X[0])):
    column = X[:,i]
    mean = np.mean(column)
    sd = np.std(column)
    infos_save.append((mean,sd))
    X_scaled[:,i] = (column-mean)/sd


X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scaled, Y, test_size=0.1)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)


# model = regression_model()
model = classification_model()


model.fit(X_train, Y_train,
          batch_size=128,
          epochs=100,
          verbose=1,
          validation_data=(X_val,Y_val))


print(np.argmax(model.predict(np.array([X_test[0]])))+1)
# print(round(model.predict(np.array([X_test[0]]))[0][0]*10))
print(Y_test[0])

model.evaluate(X_test,Y_test)

print("Do you want to save this model ? y/n")
rep = input()
if rep == 'y':
    print("Input model name :")
    name = input()
    model.save(name)

    with open("model_data.json", "r") as file:
        jsonfile = json.load(file)

    jsonfile[name] = infos_save

    with open("model_data.json", "w") as outfile:
        json.dump(jsonfile, outfile)

