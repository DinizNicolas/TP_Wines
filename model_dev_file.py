import keras
from keras import models
from keras import layers
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np

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

X,Y = preprocess_for_reg(data_)
# X,Y = preprocess_for_clas(data_)

min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scaled, Y, test_size=0.1)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)


model = regression_model()
# model = classification_model()


model.fit(X_train, Y_train,
          batch_size=128,
          epochs=200,
          verbose=1,
          validation_data=(X_val,Y_val))

#model.save('wine_model.h5')
for x in model.predict(X_test)[:2]:
    print([round(k, ndigits=1) for k in x])
print(Y_test[:2])

print(model.evaluate(X_test,Y_test))

