import yfinance as yf
import streamlit as st
import pandas as pd
import pickle

st.write("""
#Sales forecasting application



""")

st.title("📊 Walmart Sales Forecasting App")
st.write("Upload your data and get predicted sales using the trained model.")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file with input data", type=["csv"])

@st.cache_resource
def load_model():
    with open("walmart_sales_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Input Data Preview")
    st.dataframe(input_df)

    try:
        predictions = model.predict(input_df)
        input_df["Predicted Sales"] = predictions
        st.write("### 📈 Predicted Output")
        st.dataframe(input_df)
    except Exception as e:
        st.error(f"Error while making predictions: {e}")
else:
    st.info("Please upload a CSV file to continue.")
