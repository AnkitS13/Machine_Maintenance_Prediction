import streamlit as st
import pandas as pd
import pickle

# Load the trained model from the pickle file
try:
    with open('Machines.pickle', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please upload or place your 'Machines.pickle' in the same directory.")
    st.stop()

# Streamlit app layout
st.title("Machine Maintenance Prediction Dashboard")
st.write("Predict whether a machine needs maintenance based on its operational features.")

# Sidebar for user inputs
st.sidebar.header("Input Features")
def user_inputs():
    # Updated: Machine Type as Low (L), Medium (M), High (H)
    Type = st.sidebar.selectbox("Machine Type (L=Low, M=Medium, H=High)", options=["L", "M", "H"])
    Air_temp = st.sidebar.slider("Air Temperature [K]", 290, 320, 300)
    Process_temp = st.sidebar.slider("Process Temperature [K]", 300, 360, 330)
    Rotational_Speed = st.sidebar.slider("Rotational Speed [rpm]", 1000, 5000, 3000)
    Torque = st.sidebar.slider("Torque [Nm]", 10, 200, 50)
    Tool_wear = st.sidebar.slider("Tool Wear [min]", 0, 300, 100)

    # Map Type to numerical values if needed
    type_mapping = {"L": 1, "M": 2, "H": 3}
    Type = type_mapping[Type]

    # Return features as a DataFrame
    data = {
        "Type": Type,
        "Air temperature [K]": Air_temp,
        "Process temperature [K]": Process_temp,
        "Rotational speed [rpm]": Rotational_Speed,  # Make sure this matches the trained model column name
        "Torque [Nm]": Torque,
        "Tool wear [min]": Tool_wear
    }
    return pd.DataFrame([data])

input_df = user_inputs()

# Display user inputs
st.subheader("Selected Features")
st.write(input_df)

# Predictions
st.subheader("Prediction Results")
if st.button("Predict"):
    # Ensure that the columns match exactly what was used in training
    # We assume that the trained model used the exact feature names below:
    expected_columns = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    # If the columns do not match, map the user inputs columns to the expected columns
    input_df.columns = expected_columns  # Ensure the column names are exactly as expected by the model

    # Now make predictions with the model
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Get the probability of "No Maintenance" (usually the first index, prob[0][0])
    no_maintenance_prob = prediction_proba[0][0] * 100  # Probability of No Maintenance
    maintenance_needed_prob = 100 - no_maintenance_prob  # Probability of Maintenance Needed

    # Display results
    #st.write(f"Prediction: **{'Maintenance Needed' if prediction[0] == 1 else 'No Maintenance'}**")
    st.write("Prediction Probabilities:")
    st.write(f"Maintenance Needed: {maintenance_needed_prob:.2f}%")
    st.write(f"No Maintenance: {no_maintenance_prob:.2f}%")



