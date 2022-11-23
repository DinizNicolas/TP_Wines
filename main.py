from tensorflow import keras
from keras import layers
from csv import reader

with open('Wines.csv','r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)

    for row in csv_reader:
