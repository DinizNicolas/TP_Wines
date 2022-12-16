from vin import *
from models import models
import json

def notation_de_vin(vin: Vin):    
    data = [vin.fixed_acidity,vin.volatile_acidity,vin.citric_acid,vin.residual_sugar,vin.chlorides,vin.free_sulfur_dioxide,vin.total_sulfur_dioxide,vin.density,vin.ph,vin.sulphates,vin.alcohol]

    prediction = models.predict_quality(data)

    return {"Prediction": str(prediction)+"/10"}


def donne_le_vin_parfait():
    #TODO stats to determine it
    return {"message": "le vin parfait"}


def donne_modele():
    #TODO download and return file model.h5
    return {"message": "mon meilleur modele"}


def donne_description_model():
    with open("../data/model_data.json", "r") as file:
        jsonfile = json.load(file)

    model_infos = jsonfile['model_infos']

    return {"Model Informations": model_infos}


def ajoute_le_vin_au_model():
    #TODO add field to data
    return {"message": "on a ajouté un vin"}


def retrain_model():
    models.train()
    return {"message": "C'est l'heure de l'entrainement"}
