import streamlit as st
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title='Retail Sales Forecasting', layout='wide', page_icon="🛒")

st.title(":green[Forecasting Retail Revenue: Anticipating Future Sales Trends]")

holiday_options = ['True', 'False']
type_options = ['A', 'B', 'C']
size_options = [151315, 202307, 37392, 205863, 34875, 202505, 70713, 155078,
                125833, 126512, 207499, 112238, 219622, 200898, 123737, 57197,
                93188, 120653, 203819, 203742, 140167, 119557, 114533, 128107,
                152513, 204184, 206302, 93638, 42988, 203750, 203007, 39690,
                158114, 103681, 39910, 184109, 155083, 196321, 41062, 118221]

# Define the widgets for user input
# Define the widgets for user input
with st.form("my_form"):
    col1, col2, col3 = st.columns([5, 2, 5])

with col1:
    st.write(' ')
    type = st.selectbox("Type", sorted(type_options), key=3)
    store = st.slider("Store (Min: 1, Max: 45)", 1.0, 45.0, step=1.0)
    dept = st.slider("Department (Min: 1, Max: 99)", 1.0, 99.0, step=1.0)
    holiday = st.selectbox("Holiday", holiday_options, key=1)

with col3:
    st.write(' ')
    size = st.selectbox("Size", size_options, key=2)
    year = st.slider("Year (Min: 2010, Max: 2030)", 2010.0, 2030.0, step=1.0)
    month = st.slider("Month (Min: 1, Max: 12)", 1.0, 12.0, step=1.0)
    week_of_year = st.slider("Week (Min: 1, Max: 48)", 1.0, 48.0, step=1.0)
    submit_button = st.form_submit_button(label="Predict")

if submit_button:

    with open(r"D:\vs_code\sales_predict\model.pkl", 'rb') as file:
        loaded_model = pickle.load(file)
    with open(r'D:\vs_code\sales_predict\scaler.pkl', 'rb') as f:
        scaler_loaded = pickle.load(f)

    holiday_bool = (holiday == 'True')
    type_bool = (type == 'True')

    user_input_array = np.array([[store, dept, size, year, month, week_of_year, holiday_bool, type_bool]])

    numeric_cols = [1, 3, 4, 5, 6]
    categorical_cols = [0, 2, 7]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop=None, sparse=False), categorical_cols)
        ])

    X_preprocessed = preprocessor.fit_transform(user_input_array)

    expected_features = X_preprocessed.shape[1]

    if expected_features != 8:
        st.write(f"Error: Expected 8 features, but got {expected_features} features.")
    else:
        # Make predictions
        prediction = loaded_model.predict(X_preprocessed)

        st.info(f"The predicted week sales is :Rs {prediction[0]}")