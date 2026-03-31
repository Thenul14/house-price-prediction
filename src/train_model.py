import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


#load data
data = pd.read_csv("data/house_prices.csv")

#drop irrelavant columns for prediction
data = data.drop(['id','date'], axis=1)

#encoding condition column
condition_order = ['Poor', 'Fair', 'Average', 'Good', 'Very Good']
condition_encoder = OrdinalEncoder(categories=[condition_order])
data['condition'] = condition_encoder.fit_transform(data[['condition']])

#encoding waterfront column
waterfront_encoder = OrdinalEncoder(categories=[['N','Y']])
data['waterfront'] = waterfront_encoder.fit_transform(data[['waterfront']])

#features
features = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "condition", "waterfront", "floors", "view", "grade", "sqft_above", "sqft_basement", "sqft_living15", "sqft_lot15", "yr_built", "yr_renovated", "zipcode", "lat", "long"]


X = data[features]
y = data["price"]

#train the best model (XGBoost)
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


model.fit(X, y)

# Save EVERYTHING needed for prediction
pickle.dump(model, open("models/model.pkl","wb"))
pickle.dump(condition_encoder, open("models/condition_encoder.pkl","wb"))
pickle.dump(waterfront_encoder, open("models/waterfront_encoder.pkl","wb"))

print("Model saved successfully!")