import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from src.predict import predict_price
import time

#load dataset
data = pd.read_csv("data/house_prices.csv")


st.set_page_config(layout="wide")

st.title("House Price Prediction Dashboard")

st.sidebar.title("About")
st.sidebar.info("""
# House Price Prediction Dashboard

This interactive dashboard leverages machine learning techniques to estimate house prices based on key property attributes such as living area, number of rooms, condition, and location.

The model is trained on a real-world housing dataset containing over 21,000 records, enabling it to capture patterns and relationships within the data.

Users can:
-  Explore data insights and visualizations  
-  Analyze model performance  
-  Predict house prices in real-time  

---

Developed by **Thenul Jayarathna Muhandiramge**
""")

#creating tabs for each section of the app
tab1, tab2, tab3 = st.tabs(["Predict Price", "Data Explorer", "Model Insights"])

#tab 1 for prediction based on user inputs using the trained model
with tab1:
    st.header("Enter house details to estimate the price.")

    # Layout
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        bedrooms = st.slider("Bedrooms",0,40,3)
        bathrooms = st.slider("Bathrooms",0.0,10.0,2.0)
        floors = st.slider("Floors",1,4,1)
        view = st.slider("View Rating",0,4,0)
        grade = st.slider("Grade",1,13,7)
        

    
        


    with col2:
        sqft_above = st.number_input("Sqft Above",0,10000,1200)
        sqft_basement = st.number_input("Sqft Basement",0,5000,0)
        yr_built = st.number_input("Year Built",1900,2025,2000)
        yr_renovated = st.number_input("Year Renovated",0,2025,0)
        zipcode = st.number_input("Zipcode",95000,99999,98178)
        
        
        

    with col3:
        sqft_living = st.number_input("Living Area (sqft_living)",1,20000,1500)
        sqft_lot = st.number_input("Lot Size (sqft_lot)",0,1651359,5000)  
        sqft_living15 = st.number_input("Sqft living15",1,10000,1200)
        sqft_lot15 = st.number_input("Sqft lot15",0,871200,0) 

    with col4:
        lat = st.number_input("Lat ",-48.0000,48.0000,47.7776)
        long = st.number_input("Long ",-123.000,-119.000,-122.328)
        condition = st.selectbox(
            "Condition",
            ['Poor','Fair','Average','Good','Very Good']
        )
        waterfront = st.selectbox(
            "Waterfront",
            ['N','Y']
        )



    
#predict price with user inputs
    if st.button("Predict Price"):
        input_data = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "condition": condition,
        "waterfront": waterfront,
        "floors": floors,
        "view": view,
        "grade": grade,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "sqft_living15": sqft_living15,
        "sqft_lot15": sqft_lot15,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "zipcode": zipcode,
        "lat": lat,
        "long": long
        }
        
        prediction = predict_price(input_data)
       

        st.success(f"Estimated House Price: **${prediction:,.2f}**")
        


with tab2:
    st.header("Dataset Overview")

    st.write("Shape of dataset:", data.shape)

    if st.checkbox("Show Raw Data"):
        st.dataframe(data.head())
    
    st.subheader("Dataset Description")
    st.dataframe(data.describe())

    #histplot about price distribution
    st.subheader("House Price Distribution")
    st.image("assets/price distribution.png")

    #correlation heatmap
    st.subheader("Correlation Heatmap")
    st.image("assets/corr heat map.png")

    #scatterplot about features vs house price must run again after changing y to house price in notebook
    st.subheader("Scatter Plots On Some Features vs House Price")


    st.subheader("Key Insights")

    st.markdown("""
    - House prices are positively skewed  
    - sqft_living has the strongest correlation with price  
    - grade and bathrooms also strongly influence price  
    - Some features show weak correlation but may still be important  
    """)



with tab3:
    #model comparison graph(r2 score)
    st.subheader("Model Comparison (R2 Scores)")
    st.image("assets/model comparison.png")

    #best model
    st.success("Best Model: XGBoost with R² = 0.872")

    #r2 score of xgboost model
    st.metric(label="XGBoost R2 Score", value=f"{87}%")
    progress = st.progress(0)
    for i in range(87):
        time.sleep(0.04)
        progress.progress(i + 1)
    
    #actual vs predicted (xgboost)
    st.subheader("Actual vs Predicted (XGBoost)")
    st.image("assets/xgb actual vs predict.png")

    #feature importance
    st.subheader("Feature Importance")
    st.image("assets/feature imp.png")

    st.markdown("""
XGBoost was selected as the best model due to its ability to capture complex nonlinear relationships.
It achieved the highest R² score and lowest prediction error among all models.
""")
    