import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Activate the conversion rules for rpy2.robjects


st.sidebar.title('Simple Data Projects')
page = st.sidebar.radio(
    "",
    ("SALARY PREDICTION",
     "WEIGHT PREDICTION")
)

if page == "SALARY PREDICTION":
    
    st.markdown("#### SALARY PREDICTION SYSTEM")
    st.markdown("")
    st.markdown("")
    salary_data_df = pd.read_csv("Salary_Data.csv")
    work_exp = st.slider("Choose your Working Experience",min_value=float(0.0), max_value=float(12.0))

    X = salary_data_df.iloc[:, :-1].values
    y = salary_data_df.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

    # Training the Simple Linear Regression model on the Training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results using Python
    y_pred = regressor.predict(X_test)

    pred = round(regressor.predict([[work_exp]])[0],2)
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("##### Predicted Salary using Python Linear Regression")
    st.write('₹',str(pred),'per annum(Python)')
    
    # Predicting the Test set results using R
    
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("##### Predicted Salary using R Linear Regression")
    st.write('₹',str(pred),'per annum')

if page == "WEIGHT PREDICTION":
    st.markdown("#### WEIGHT PREDICTION SYSTEM")
    st.markdown("")
    st.markdown("")
    weight_height_df = pd.read_csv("weight-height.csv")
    height = st.slider("Choose your Height",min_value=float(50.0), max_value=float(80.0))
    
    X = weight_height_df.iloc[:, :-1].values
    y = weight_height_df.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

    # Training the Simple Linear Regression model on the Training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = regressor.predict(X_test)

    pred = round(regressor.predict([[height]])[0],2)
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("##### Predicted Weight")
    st.write(str(pred),'KG')