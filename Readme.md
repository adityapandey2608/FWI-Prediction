# 🔥 Fire Weather Index (FWI) Prediction App

This is a Flask web application that predicts the **Fire Weather Index (FWI)** using a machine learning model trained on the **Algerian Forest Fire dataset**.

---

## 📊 Dataset Info

- **Source**: [Algerian Forest Fire Dataset](https://archive.ics.uci.edu/ml/datasets/Algerian+Forest+Fires+Dataset+)
- **Attributes used**:
  - Temperature
  - Relative Humidity (RH)
  - Wind Speed (Ws)
  - Rain
  - FFMC (Fine Fuel Moisture Code)
  - DMC (Duff Moisture Code)
  - ISI (Initial Spread Index)
  - Class (Fire occurrence class)

The dataset includes meteorological and fire data from two regions of Algeria collected between June and September 2012.

---

## 🧠 Model Training

The model was trained using **Ridge Regression** in scikit-learn. The steps included:

1. Cleaning the dataset (removing nulls, standardizing column names)
2. Encoding categorical variables if needed
3. Scaling features using `StandardScaler`
4. Training a Ridge Regression model on the scaled data
5. Saving the model (`regressor.pkl`) and scaler (`scaler.pkl`) using `pickle`

> The training process and data exploration are documented in the `Model Training.ipynb` and `Fire Forest EDA.ipynb` notebooks.

---

## 🚀 Features

- Simple web interface for FWI prediction
- Scikit-learn Ridge Regression model
- Input form for environmental variables
- Real-time prediction result
- Scaled inputs using `StandardScaler`
- Clean user interface with HTML/CSS

---

## 🛠 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/fwi-predictor.git
cd fwi-predictor
