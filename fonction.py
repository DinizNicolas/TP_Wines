from vin import *
from models import models
import json
from csv import writer

def note_wine(vin: Vin):    
    data = [vin.fixed_acidity,vin.volatile_acidity,vin.citric_acid,vin.residual_sugar,vin.chlorides,vin.free_sulfur_dioxide,vin.total_sulfur_dioxide,vin.density,vin.ph,vin.sulphates,vin.alcohol]

    prediction = models.predict_quality(data)

    return {"Prediction": str(prediction)+"/10"}


def give_perfect_wine():
    #TODO stats to determine it
    return {"message": "le vin parfait"}


def give_model():
    #TODO download and return file model.h5
    return {"message": "mon meilleur modele"}


def give_model_description():
    with open("../data/model_data.json", "r") as file:
        jsonfile = json.load(file)

    model_infos = jsonfile['model_infos']

    return {"Model Informations": model_infos}


def add_wine_to_dataset(vin: Vin):
    data = [vin.fixed_acidity,vin.volatile_acidity,vin.citric_acid,vin.residual_sugar,vin.chlorides,vin.free_sulfur_dioxide,vin.total_sulfur_dioxide,vin.density,vin.ph,vin.sulphates,vin.alcohol,vin.quality,vin.id]
    
    with open("../data/Wines.csv",'a') as file:

        writer_object = writer(file)
        writer_object.writerow(data)
        file.close()

    return {"message": "Wine added"}


def retrain_model():
    models.train()
    return {"message": "Model Retrained"}
