from enum import Enum
from fastapi import FastAPI
from fonction import *

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Get me some wine"}


@app.post("/api/predict")
async def note_ce_vin(vin: Vin):
    return notation_de_vin(vin)


@app.get("/api/predict")
async def donne_vin_parfait():
    return donne_le_vin_parfait();


@app.get("/api/model")
async def donne_le_modele():
    return donne_modele()


@app.get("/api/model/description")
async def donne_la_description_du_model():
    return donne_description_model();

@app.put("/api/model")
async def ajoute_un_vin(vin: Vin):
    return ajoute_le_vin_au_model(vin);


@app.post("/api/model/retrain")
async def entraine_le_model():
    return retrain_model();



