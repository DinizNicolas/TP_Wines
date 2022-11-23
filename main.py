from tensorflow import keras
from keras import layers
from csv import reader

with open('Wines.csv','r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    inputs_ = []
    outputs_ = []

    for row in csv_reader:
        print(row)
        list_values_tmp = [float(value) for value in row]
        outputs_.append(list_values_tmp[-2])
        inputs_.append(list_values_tmp[0:-2])

