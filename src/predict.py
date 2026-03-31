import numpy as np
import pickle

model = pickle.load(open("models/model.pkl", "rb"))
encoder_condition = pickle.load(open("models/condition_encoder.pkl", "rb"))
encoder_waterfront = pickle.load(open("models/waterfront_encoder.pkl", "rb"))

def predict_price(data):
    condition_encoded = encoder_condition.transform([[data["condition"]]])[0][0]
    waterfront_encoded = encoder_waterfront.transform([[data["waterfront"]]])[0][0]

    input_data = np.array([[ 
        data["bedrooms"],
        data["bathrooms"],
        data["sqft_living"],
        data["sqft_lot"],
        condition_encoded,
        waterfront_encoded,
        data["floors"],
        data["view"],
        data["grade"],
        data["sqft_above"],
        data["sqft_basement"],
        data["sqft_living15"],
        data["sqft_lot15"],
        data["yr_built"],
        data["yr_renovated"],
        data["zipcode"],
        data["lat"],
        data["long"]
    ]])

    return model.predict(input_data)[0]