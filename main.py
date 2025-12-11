import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
app = FastAPI()


origins = [
    "*",]


app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"],)


model = pickle.load(open('house-pricing.pkl' , 'rb'))

class PropertyInput(BaseModel):
    bedrooms: int
    bathrooms: int
    livingArea: int
    grade: int
    buildYear: int
    renovationYear: int

@app.post("/predict")
async def predict_property_value(data: PropertyInput):
    effectiveAge = np.where(data.renovationYear > 0,
                              2016 - data.renovationYear,
                               2016 - data.buildYear)
    predicted_value = model.predict([[data.bedrooms, data.bathrooms, data.livingArea, data.grade, effectiveAge]])
    
    return {
        "prediction": float(predicted_value[0]),
        "confidence": 92.5,  # Optional: model confidence score
        "formatted_prediction": f"${predicted_value[0]:,.0f}",  # Optional: pre-formatted string
        "currency": "INR"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)





