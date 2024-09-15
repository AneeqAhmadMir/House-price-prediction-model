import numpy as np
import pandas as pd
import pickle as pk
import streamlit as st

model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))  # Load the scaler

st.header('House Price Prediction ML Model')

house_data = pd.read_csv('USA_Housing.csv')

Avg_Area_Income = st.slider('Select Avg. Area Income', 17796.631190, 107701.748378)
Avg_Area_House_Age = st.slider('Select Avg. Area House Age', 2.644304, 9.519088)
Avg_Area_Number_of_Rooms = st.slider('Select Avg. Area Number of Rooms', 3.236194, 10.759588)
Avg_Area_Number_of_Bedrooms = st.slider('Select Avg. Area Number of Bedrooms', 2.000000, 6.500000)
Area_Population = st.slider('Select Area Population', 172.610686, 69621.713378)

if st.button('Predict'):
    inputs = np.array([[Avg_Area_Income, Avg_Area_House_Age, Avg_Area_Number_of_Rooms,
                        Avg_Area_Number_of_Bedrooms, Area_Population]])

    inputs_scaled = scaler.transform(inputs)
    house_price = model.predict(inputs_scaled)
    st.markdown('House Price is going to be $' + str(house_price[0]))
