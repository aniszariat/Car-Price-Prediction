from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from datetime import datetime
import traceback

# initialize the FastAPI app
app = FastAPI()


with open("../models/randomForestModel.pkl", "rb") as f:
    model = pickle.load(f)

with open("../models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("../models/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("../models/one_hot_encoder.pkl", "rb") as f:
    one_hot_encoder = pickle.load(f)


class CarInput(BaseModel):
    Levy: int
    Manufacturer: str
    Model: str
    Prod_year: int
    Category: str
    Leather_interior: str
    Fuel_type: str
    Engine_volume: float
    Mileage: int
    Cylinders: int
    Gear_box_type: str
    Drive_wheels: str
    Wheel: str
    Color: str
    Airbags: int


@app.get("/")
def read_root():
    return {"message": "Welcome to the Car Price Prediction API!"}


@app.post("/predict/")
def predict(car_data: CarInput):
    try:
        data = pd.DataFrame([car_data.dict()])
        data["Age"] = datetime.now().year - data["Prod_year"]
        # drop unsed columns
        data = data.drop(columns=["Doors", "Prod_year"], errors="ignore")

        # Rename columns to match the preprocessing pipeline
        column_rename_map = {
            "Leather_interior": "Leather interior",
            "Gear_box_type": "Gear box type",
            "Drive_wheels": "Drive wheels",
            "Engine_volume": "Engine volume",
            "Fuel_type": "Fuel type",
        }
        data.rename(columns=column_rename_map, inplace=True)
        # One-hot encode categorical columns
        one_hot_columns = ["Leather interior", "Gear box type", "Drive wheels", "Wheel"]
        encoded_data = one_hot_encoder.transform(data[one_hot_columns])
        encoded_data_df = pd.DataFrame(
            encoded_data,
            columns=one_hot_encoder.get_feature_names_out(one_hot_columns),
            index=data.index,
        )
        # Concatenate the one-hot encoded data and drop original columns
        data = pd.concat([data, encoded_data_df], axis=1)
        data.drop(columns=one_hot_columns, inplace=True)

        # Label encode categorical columns
        label_encode_columns = [
            "Manufacturer",
            "Model",
            "Category",
            "Fuel type",
            "Color",
        ]
        for col in label_encode_columns:
            if col in data.columns:
                data[col] = label_encoders[col].transform(data[col])

        # Scale numerical columns
        numerical_columns = ["Levy", "Engine volume", "Mileage", "Age"]
        data[numerical_columns] = scaler.transform(data[numerical_columns])
        # Make prediction
        prediction = model.predict(data)
        return {"predicted": prediction[0]}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
