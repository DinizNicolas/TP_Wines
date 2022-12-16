import numpy as np
import pandas as pd
import json
from sklearn import preprocessing as pre
from sklearn.model_selection import train_test_split


'''func preprocess
_inputs : 
    data_ : matrix of size (unknown,13), containing floats
_outputs :
    X : matrix of size (unknown,11), the first 11 columns
    Y : list of size (unknown), the columns before last, containing values between 0 and 1
'''
def preprocess(data_):
    X = data_[:,0:11]
    Y = data_[:,-2]
    Y = Y/10
    return (X,Y)

'''func load_csv
_inputs : 
    filename : string, the path of the file
_outputs :
    data : panda dataframe, containing the csv data
'''
def load_csv(filename: str):
    if '.csv' not in filename:
        filename += '.csv'
    data = pd.read_csv(filename)

    return data


'''func save_scaling_info
_inputs : 
    matrix of int, the data before being scaled
_outputs :
    // scaling infos saved into a json file 
'''
def save_scaling_info(data):
    infos_save = []
    for i in range(len(data[0])):
        column = data[:,i]
        mean = np.mean(column)
        sd = np.std(column)
        infos_save.append((mean,sd))

    with open("../data/model_data.json", "w") as outfile:
        json.dump(infos_save, outfile)

    return 0

'''func scale_data
_inputs : 
    data_ : matrix of size (unknown,11), containing floats
_outputs :
    data_scaled : the input data scaled using the formula (data - mean)/standart_deviation
'''
def scale_data(data):
    min_max_scaler = pre.StandardScaler()
    data_scaled = min_max_scaler.fit_transform(data)

    save_scaling_info(data)

    return data_scaled

'''func split_data
_inputs : 
    X : matrix of size (unknown,11), containing floats between -1 and 1
    Y : list of elements (float or list)
_outputs :
    [X_train,Y_train,X_val,Y_val,X_test,Y_test] : input data split into training data, validation data and test data
'''
def split_data(X,Y):
    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, Y, test_size=0.1)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

    return [X_train,Y_train,X_val,Y_val,X_test,Y_test]

'''func preprocessing
_inputs : 
    data_filename : string, path to the csv file containing the data
_outputs :
    [X_train,Y_train,X_val,Y_val,X_test,Y_test] : data preprocessed for the model selected, split into training data, validation data and test data
'''
def preprocessing(data_filename: str):
    data = load_csv('../data/'+data_filename)

    #Change values type to float
    for name in data.columns:
        data[name] = data[name].astype(float)
    
    #Retrieve matrix
    data = data.values

    X,Y = None,None
    X,Y = preprocess(data)

    X_scaled = scale_data(X)
    [X_train,Y_train,X_val,Y_val,X_test,Y_test] = split_data(X_scaled,Y)

    return [X_train,Y_train,X_val,Y_val,X_test,Y_test]