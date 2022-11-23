import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def preprocess_for_regression(data_):
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

def load_csv(filename: str):
    if '.csv' not in filename:
        filename += '.csv'
    data = pd.read_csv(filename)

    return data

def scale_data(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)

    return data_scaled

def split_data(X,Y):
    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, Y, test_size=0.1)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

    return [X_train,Y_train,X_val,Y_val,X_test,Y_test]

def preprocessing(modeltype: str,data_filename: str):
    data = load_csv(data_filename)

    #Change values type to float
    for name in data.columns:
        data[name] = data[name].astype(float)
    
    #Retrieve matrix
    data = data.value

    X,Y = None,None
    if modeltype == 'classification':
        X,Y = preprocess_for_clas(data)
    elif modeltype == 'regression':
        X,Y = preprocess_for_regression(data)

    X_scaled = scale_data(X)
    [X_train,Y_train,X_val,Y_val,X_test,Y_test] = split_data(X_scaled,Y)

    return [X_train,Y_train,X_val,Y_val,X_test,Y_test]