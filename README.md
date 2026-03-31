# House Price Prediction Dashboard

## Project Overview
This project focuses on predicting house prices using Machine Learning models and presenting the results through an interactive **Streamlit dashboard**.

The application allows users to:
- Explore the dataset (EDA)
- Understand model performance
- Predict house prices based on user input

The goal is to simulate a **real-world end-to-end ML project**, from data analysis to deployment.

---

## Dataset
The dataset used is the **House Pricing Dataset** (Kaggle), containing:

- **21,613 records**
- **21 features**

Key features include:
- Bedrooms & Bathrooms
- Square footage (living area & lot)
- Property condition & grade
- Year built / renovated
- Location-related features

---

## Exploratory Data Analysis (EDA)

Key findings:

- No missing values or duplicates
- House prices are **positively skewed**
- Strong correlations with price:
  - `sqft_living`
  - `grade`
  - `sqft_above`
- Weak correlations do not imply irrelevance, as ML models capture nonlinear relationships

Visualizations included:
- Distribution plots
- Correlation heatmap
- Feature vs Price scatter plots

---

## Models Used

The following models were trained and compared:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor ⭐ (Best)

---

## Model Performance

| Model | RMSE | MAE | R² Score |
|------|------|------|---------|
| Linear Regression | 212,539 | 127,493 | 0.701 |
| Random Forest | 152,321 | 75,684 | 0.847 |
| Gradient Boosting | 141,249 | 69,755 | 0.868 |
| XGBoost  | **139,195** | **68,322** | **0.872** |

**XGBoost achieved the best performance**, making it the final model used in the application.

---

##  Streamlit Dashboard

The project includes an interactive dashboard built with **Streamlit**.

### Features:
- User-friendly input interface
- Real-time price prediction
- Data visualizations (EDA)
- Model performance insights

---

##  How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/Thenul14/house-price-prediction.git
cd house-price-prediction
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the Streamlit app
```bash
streamlit run app.py
```

## Project Structure
```bash
house-price-prediction/
│
├── data/
│   └── house_prices.csv
│
├── models/
│   ├── model.pkl
│   ├── condition_encoder.pkl
│   └── waterfront_encoder.pkl
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── train_model.py
│   └── predict.py
│
├── app.py
├── requirements.txt
└── README.md
```

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib & Seaborn
- Streamlit

## Key Features

- End-to-end ML pipeline
- Hyperparameter tuning (GridSearch & RandomizedSearch)
- Model comparison and evaluation
- Interactive dashboard
- Feature importance analysis

## Limitations

- Some location features may not fully capture real-world geographic effects
- Model performance may vary on unseen datasets
- Limited feature engineering

## Future Improvements

- Add SHAP for model explainability
- Deploy app online (Streamlit Cloud / Render)
- Improve UI/UX design
- Use additional datasets for better generalization


## Author

**Thenul Chamikara Jayarathna Muhandiramge**

## Support

If you found this project useful, consider giving it a ⭐ on GitHub!


---

