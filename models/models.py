from keras import models
from keras import layers
from keras import metrics
import numpy as np
from vin import Vin
import sys
import json
from preprocessing import preprocessing

#Adding project root to path
sys.path.append("../")

MODEL_NAME = "model.h5"

def regression_model():
    model = models.Sequential()
    # Model structure
    model.add(layers.Dense(11, input_shape=(11,), kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(11, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(1, kernel_initializer='normal', activation='linear'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=[metrics.MeanAbsolutePercentageError()])

    return model

def load_model():
    model = None
    model = models.load_model(MODEL_NAME)

    return model

def save_model(model):
    model.save(MODEL_NAME)

def save_model_infos(model,X,Y,):
    metrics_names = model.metrics_names
    metrics_values = model.evaluate(X,Y)
    model_config = model.get_config()

    with open("../data/model_data.json", "r") as file:
        jsonfile = json.load(file)
        file.close()

    jsonfile["model_infos"]["metrics"] = (metrics_names,metrics_values)
    jsonfile["model_infos"]["config"] = model_config

    with open("../data/model_data.json", "w") as outfile:
        json.dump(jsonfile, outfile)
        outfile.close()

def train():
    BATCH_SIZE = 128
    EPOCHS = 100
    [X_train,Y_train,X_val,Y_val,X_test,Y_test] = preprocessing("Wines.csv")

    model = regression_model()

    model.fit(X_train, Y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            validation_data=(X_val,Y_val))

    model.evaluate(X_test,Y_test)

    save_model_infos(model,X_test,Y_test)
    save_model(model)

def predict_quality(data: list):
    model = load_model()

    #preprocss data correctly depending on model type
    with open("../data/model_data.json", "r") as file:
        jsonfile = json.load(file)
        file.close()

    data_preprocess = jsonfile["scaling_data"]

    for i in range(len(data)):
        (mean,sd) = data_preprocess[i]
        data[i] = (data[i]-mean)/sd

    
    return round(model.predict(np.array([data]))[0][0]*10)
