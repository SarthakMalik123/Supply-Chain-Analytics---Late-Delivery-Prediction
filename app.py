import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------
# Load model & encoders
# -------------------
with open("best_classifier.pkl", "rb") as file:
    model = pickle.load(file)

with open("encoders.pkl", "rb") as file:
    encoders = pickle.load(file)  # dict or list of fitted LabelEncoders

st.title("📦 Delivery Risk Prediction App")
st.write("Enter the shipment/order details below:")

col1, col2 = st.columns(2)

with col1:
    type_ = st.selectbox("Type", ["DEBIT", "TRANSFER", "CASH", "PAYMENT"])
    days_real = st.number_input("Days for shipping (real)", 0, 30, 3)
    days_scheduled = st.number_input("Days for shipment (scheduled)", 0, 30, 4)
    delivery_status = st.selectbox("Delivery Status", ["Advance shipping", "Late delivery", "Shipping on time"])
    category_name = st.selectbox("Category Name", ["Sporting Goods", "Technology", "Office Supplies", "Furniture"])
    customer_city = st.text_input("Customer City", "Caguas")
    customer_lname = st.text_input("Customer Lname", "Luna")
    customer_segment = st.selectbox("Customer Segment", ["Consumer", "Corporate", "Home Office", "Small Business"])
    customer_zipcode = st.number_input("Customer Zipcode", 0, 99999, 725)

with col2:
    product_price = st.number_input("Product Price", 0.0, 100000.0, 327.75)
    shipping_mode = st.selectbox("Shipping Mode", ["Standard Class", "Second Class", "First Class", "Same Day"])
    ship_year = st.number_input("Shipping Year", 2000, 2030, 2020)
    ship_month = st.number_input("Shipping Month", 1, 12, 1)
    ship_day = st.number_input("Shipping Day", 1, 31, 15)
    ship_hour = st.number_input("Shipping Hour", 0, 23, 11)
    order_year = st.number_input("Order Year", 2000, 2030, 2018)
    order_month = st.number_input("Order Month", 1, 12, 1)
    order_day = st.number_input("Order Day", 1, 31, 13)
    order_hour = st.number_input("Order Hour", 0, 23, 12)

if st.button("Predict"):
    # Arrange input data
    input_data = [
        type_,
        days_real,
        days_scheduled,
        delivery_status,
        category_name,
        customer_city,
        customer_lname,
        customer_segment,
        customer_zipcode,
        product_price,
        shipping_mode,
        ship_year,
        ship_month,
        ship_day,
        ship_hour,
        order_year,
        order_month,
        order_day,
        order_hour
    ]

    # Encode categorical features (indices based on your training order)
    categorical_indices = [0, 3, 4, 5, 6, 7, 10]  # adjust if needed
    for idx in categorical_indices:
        input_data[idx] = encoders[idx].transform([input_data[idx]])[0]

    # Convert to NumPy array
    input_array = np.array(input_data, dtype=float).reshape(1, -1)

    # Predict
    prediction = model.predict(input_array)
    st.success(f"🧠 Predicted Outcome: {prediction[0]}")
