import keras
from keras import models
from keras import layers
# from csv import reader
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('Wines.csv')
for name in data.columns:
    data[name] = data[name].astype(float)

data_ = data.values
print(data.dtypes)

X = data_[:,0:11]
Y = data_[:,-2]
Y = Y/10
print(Y)

min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scaled, Y, test_size=0.1)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

model = models.Sequential()
model.add(layers.Dense(11, input_shape=(11,), kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(4, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(1, kernel_initializer='normal', activation='linear'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam',metrics=[keras.metrics.MeanAbsolutePercentageError(),keras.metrics.MeanAbsoluteError()])


model.fit(X_train, Y_train,
          batch_size=128,
          epochs=100,
          verbose=1,
          validation_data=(X_test,Y_test))

print([x[0] for x in model.predict(X_test)[:5]])
print(Y_test[:5])

print(model.evaluate(X_test,Y_test))
