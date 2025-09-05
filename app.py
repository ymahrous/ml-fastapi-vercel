import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("car-recommender.joblib")
app = FastAPI()


class CarUser(BaseModel):
    age: int
    gender: int


@app.post("/car/predict")
async def recommend_car(features: CarUser):

    recommendation = model.predict([[features.age, features.gender]])
    return {"pred": recommendation[0]}


@app.get("/")
async def root():
    return {"message": "Welcome to the Car Recommender API!"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
