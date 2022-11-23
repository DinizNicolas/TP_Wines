from keras import models
from keras import layers
from keras import metrics
import numpy as np
from vin import Vin


def regression_model():
    model = models.Sequential()

    model.add(layers.Dense(11, input_shape=(11,), kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(11, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(1, kernel_initializer='normal', activation='linear'))
    
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=[metrics.MeanAbsolutePercentageError()])

    return model

def classification_model():
    model = models.Sequential()

    model.add(layers.Dense(64, input_shape=(11,), kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(10, kernel_initializer='normal', activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def load_model(modeltype: str):
    model = None
    if modeltype == 'classification':
        model = models.load_model('wine_clas_model.h5')
    elif modeltype == 'regression':
        model = models.load_model('wine_reg_model.h5')

    return model

def predict_quality(modeltype: str,data):
    model = load_model(modeltype)

    if modeltype == 'classification':
        return np.argmax(model.predict(np.array([data])))+1
    elif modeltype == 'regression':
        return round(model.predict(np.array([data]))[0][0]*10)
