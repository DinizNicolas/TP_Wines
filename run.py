from fastapi import FastAPI
from fonction import *

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Get me some wine"}


@app.post("/api/predict")
async def wine_notation(vin: Vin):
    return note_wine(vin)


@app.get("/api/predict")
async def give_the_perfect_wine():
    return give_perfect_wine();


@app.get("/api/model")
async def give_the_model():
    return give_model()


@app.get("/api/model/description")
async def give_description_of_model():
    return give_model_description();

@app.put("/api/model")
async def add_wine(vin: Vin):
    return add_wine_to_dataset(vin);


@app.post("/api/model/retrain")
async def train_model():
    return retrain_model();



