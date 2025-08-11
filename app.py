#app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------
# Load model, scaler, encoders, and columns
# -------------------
try:
    with open("best_classifier.pkl", "rb") as file:
        model = pickle.load(file)
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    with open("encoders.pkl", "rb") as file:
        encoders = pickle.load(file)
    with open("final_columns.pkl", "rb") as file:
        final_columns = pickle.load(file)
except FileNotFoundError:
    st.error("Model or preprocessing files not found. Please run the notebook first to generate them.")
    st.stop()


st.title("📦 Supply Chain Late Delivery Risk Prediction")
st.write("Enter the shipment and order details below to predict the risk of late delivery.")

# --- Create input fields for all features used by the model ---
# Use columns for better layout
col1, col2, col3 = st.columns(3)

# --- User Inputs ---
with col1:
    st.subheader("Shipment Details")
    days_real = st.number_input("Days for shipping (real)", 0, 30, 3)
    days_scheduled = st.number_input("Days for shipment (scheduled)", 0, 30, 4)
    shipping_mode = st.selectbox("Shipping Mode", ['Standard Class', 'Second Class', 'First Class', 'Same Day'])
    market = st.selectbox("Market", ['LATAM', 'Europe', 'Pacific Asia', 'USCA', 'Africa'])
    
    st.subheader("Location Details")
    latitude = st.number_input("Latitude", -90.0, 90.0, 18.25)
    longitude = st.number_input("Longitude", -180.0, 180.0, -66.0)
    customer_zipcode = st.number_input("Customer Zipcode", 0, 99999, 725)


with col2:
    st.subheader("Order & Product Details")
    type_ = st.selectbox("Type", ["DEBIT", "TRANSFER", "CASH", "PAYMENT"])
    category_name = st.selectbox("Category Name", ['Sporting Goods', 'Cleats', 'Shop By Sport', "Men's Footwear", 'Electronics']) # Example values
    product_price = st.number_input("Product Price", 0.0, 100000.0, 327.75)
    order_item_total = st.number_input("Order Item Total", 0.0, 100000.0, 314.64)
    order_profit_per_order = st.number_input("Order Profit Per Order", -5000.0, 5000.0, 91.25)
    order_item_discount_rate = st.slider("Order Item Discount Rate", 0.0, 1.0, 0.04)
    order_item_profit_ratio = st.slider("Order Item Profit Ratio", -3.0, 3.0, 0.29)
    order_item_quantity = st.number_input("Order Item Quantity", 1, 100, 1)


with col3:
    st.subheader("Customer & Order Status")
    customer_segment = st.selectbox("Customer Segment", ["Consumer", "Corporate", "Home Office"])
    department_name = st.selectbox("Department Name", ['Fitness', 'Apparel', 'Golf', 'Footwear', 'Outdoors']) # Example values
    order_status = st.selectbox("Order Status", ['COMPLETE', 'PENDING', 'CLOSED', 'CANCELED', 'PENDING_PAYMENT']) # Example values
    order_region = st.selectbox("Order Region", ['South America', 'Southeast Asia', 'Central America', 'Oceania', 'Western Europe']) # Example values

    # For other high cardinality features, text input is practical for a demo
    customer_city = st.text_input("Customer City", "Caguas")
    customer_lname = st.text_input("Customer Lname", "Luna")
    order_country = st.text_input("Order Country", "Puerto Rico")
    order_state = st.text_input("Order State", "PR")
    product_name = st.text_input("Product Name", "Smart watch")
    order_city = st.text_input("Order City", "Caguas")
    
# --- Add remaining date features which were not dropped ---
# From your notebook, ship_month, ship_day, order_day, order_hour were used.
# Let's add them for completeness.
    st.subheader("Remaining Date Features")
    ship_month = st.number_input("Shipping Month", 1, 12, 2)
    ship_day = st.number_input("Shipping Day", 1, 31, 3)
    order_day = st.number_input("Order Day", 1, 31, 31)
    
if st.button("Predict Delivery Risk"):
    
    # --- 1. Create a DataFrame from inputs ---
    # The dictionary keys must match the column names from your notebook's `X` dataframe
    input_dict = {
        'Days for shipping (real)': days_real,
        'Days for shipment (scheduled)': days_scheduled,
        'Customer Zipcode': customer_zipcode,
        'Latitude': latitude,
        'Longitude': longitude,
        'Order Customer Id': 1, # Placeholder
        'Order Id': 1, # Placeholder
        'Order Item Discount Rate': order_item_discount_rate,
        'Order Item Profit Ratio': order_item_profit_ratio,
        'Order Item Quantity': order_item_quantity,
        'Order Item Total': order_item_total,
        'Order Profit Per Order': order_profit_per_order,
        'Product Price': product_price,
        'ship_month': ship_month,
        'ship_day': ship_day,
        'order_day': order_day,
        'Type': type_,
        'Delivery Status': 'Advance shipping', # Placeholder, as it gets dropped
        'Category Name': category_name,
        'Customer City': customer_city,
        'Customer Lname': customer_lname,
        'Customer Segment': customer_segment,
        'Department Name': department_name,
        'Market': market,
        'Order City': order_city,
        'Order Country': order_country,
        'Order Region': order_region,
        'Order State': order_state,
        'Order Status': order_status,
        'Product Name': product_name,
        'Shipping Mode': shipping_mode
    }
    input_df = pd.DataFrame([input_dict])

    # --- 2. Preprocess the DataFrame ---
    
    # One-Hot Encode low cardinality features
    one_hot_cols = ['Customer Segment', 'Type', 'Shipping Mode', 'Market'] # 'Delivery Status' was dropped
    input_df = pd.get_dummies(input_df, columns=one_hot_cols, drop_first=False)

    # Label Encode high cardinality features using saved encoders
    for col, le in encoders.items():
        # Handle unseen labels by assigning a default value (e.g., -1 or a specific known category)
        input_df[col] = input_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # --- 3. Align columns with the training data ---
    # Ensure all columns from training are present and in the correct order
    input_df_aligned = input_df.reindex(columns=final_columns, fill_value=0)
    
    # Drop the delivery status columns as done in the notebook
    input_df_aligned = input_df_aligned.loc[:, ~input_df_aligned.columns.str.startswith('Delivery')]

    # --- 4. Scale the data ---
    input_scaled = scaler.transform(input_df_aligned)

    # --- 5. Predict ---
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # --- Display Results ---
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.warning(f"**Risk of Late Delivery: YES** (Probability: {prediction_proba[0][1]:.2f})")
    else:
        st.success(f"**Risk of Late Delivery: NO** (Probability: {prediction_proba[0][0]:.2f})")
