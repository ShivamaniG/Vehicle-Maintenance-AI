import streamlit as st
import pickle
import numpy as np
import shap

# Load the trained model (replace 'model.pkl' with your actual file name)
with open('hhmodel.pkl', 'rb') as file:
    model = pickle.load(file)

# Load dataset for SHAP explanation (for demonstration purposes, we use the California dataset)
X, y = shap.datasets.california()

# Explain the model's predictions using SHAP
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Define the customized ranges for each feature based on dataset statistics
custom_ranges = {
    'Engine rpm': (61.0, 2239.0),
    'Lub oil pressure': (0.003384, 7.265566),
    'Fuel pressure': (0.003187, 21.138326),
    'Coolant pressure': (0.002483, 7.478505),
    'lub oil temp': (71.321974, 89.580796),
    'Coolant temp': (61.673325, 195.527912),
    'Temperature_difference': (-22.669427, 119.008526)
}

# Minimum values for vehicle to run
min_conditions = {
    'Engine rpm': 500,
    'Lub oil pressure': 0.2,
    'Fuel pressure': 0.5,
    'Coolant pressure': 0.5,
    'lub oil temp': 75,
    'Coolant temp': 70,
    'Temperature_difference': 0
}

# Feature Descriptions
feature_descriptions = {
    'Engine rpm': 'Revolution per minute of the engine.',
    'Lub oil pressure': 'Pressure of the lubricating oil.',
    'Fuel pressure': 'Pressure of the fuel.',
    'Coolant pressure': 'Pressure of the coolant.',
    'lub oil temp': 'Temperature of the lubricating oil.',
    'Coolant temp': 'Temperature of the coolant.',
    'Temperature_difference': 'Temperature difference between components.'
}

# Maintenance Days (based on condition)
maintenance_days = {
    'Engine rpm': 30,  # Example maintenance days if rpm is low
    'Lub oil pressure': 7,  # Example maintenance days if oil pressure is low
    'Fuel pressure': 14,  # Example maintenance days if fuel pressure is low
    'Coolant pressure': 10,  # Example maintenance days if coolant pressure is low
    'lub oil temp': 5,  # Example maintenance days if oil temp is high
    'Coolant temp': 7,  # Example maintenance days if coolant temp is high
    'Temperature_difference': 3  # Example maintenance days for high temp difference
}

# Engine Condition Prediction App
def main():
    st.title("Engine Condition Prediction")

    # Display feature descriptions
    st.sidebar.title("Feature Descriptions")
    for feature, description in feature_descriptions.items():
        st.sidebar.markdown(f"**{feature}:** {description}")

    # Display minimum conditions to run
    st.sidebar.title("Minimum Conditions to Run")
    for feature, min_value in min_conditions.items():
        st.sidebar.markdown(f"**{feature}:** {min_value}")

    # Input widgets with customized ranges
    engine_rpm = st.slider("Engine RPM", min_value=float(custom_ranges['Engine rpm'][0]), 
                           max_value=float(custom_ranges['Engine rpm'][1]), 
                           value=float(custom_ranges['Engine rpm'][1] / 2))
    lub_oil_pressure = st.slider("Lub Oil Pressure", min_value=custom_ranges['Lub oil pressure'][0], 
                                 max_value=custom_ranges['Lub oil pressure'][1], 
                                 value=(custom_ranges['Lub oil pressure'][0] + custom_ranges['Lub oil pressure'][1]) / 2)
    fuel_pressure = st.slider("Fuel Pressure", min_value=custom_ranges['Fuel pressure'][0], 
                              max_value=custom_ranges['Fuel pressure'][1], 
                              value=(custom_ranges['Fuel pressure'][0] + custom_ranges['Fuel pressure'][1]) / 2)
    coolant_pressure = st.slider("Coolant Pressure", min_value=custom_ranges['Coolant pressure'][0], 
                                 max_value=custom_ranges['Coolant pressure'][1], 
                                 value=(custom_ranges['Coolant pressure'][0] + custom_ranges['Coolant pressure'][1]) / 2)
    lub_oil_temp = st.slider("Lub Oil Temperature", min_value=custom_ranges['lub oil temp'][0], 
                             max_value=custom_ranges['lub oil temp'][1], 
                             value=(custom_ranges['lub oil temp'][0] + custom_ranges['lub oil temp'][1]) / 2)
    coolant_temp = st.slider("Coolant Temperature", min_value=custom_ranges['Coolant temp'][0], 
                             max_value=custom_ranges['Coolant temp'][1], 
                             value=(custom_ranges['Coolant temp'][0] + custom_ranges['Coolant temp'][1]) / 2)
    temp_difference = st.slider("Temperature Difference", min_value=custom_ranges['Temperature_difference'][0], 
                                max_value=custom_ranges['Temperature_difference'][1], 
                                value=(custom_ranges['Temperature_difference'][0] + custom_ranges['Temperature_difference'][1]) / 2)

    # Predict button
    if st.button("Predict Engine Condition"):
        result, confidence = predict_condition(engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference)
        
        # Show Maintenance Days Required
        st.subheader("Maintenance Days Required")
        for feature, value in zip(custom_ranges.keys(), [engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference]):
            if value < min_conditions[feature]:
                st.warning(f"{feature} is below minimum condition! Maintenance required in {maintenance_days[feature]} days.")
            else:
                st.success(f"{feature} is above the minimum condition.")
        
        st.subheader("Model Explanation (SHAP)")

        # Get SHAP explanation for user input
        input_data = np.array([engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference]).reshape(1, -1)
        shap_values = explainer(input_data)  # Get SHAP values for the input

        # Display SHAP plot
        st.pyplot(shap.plots.waterfall(shap_values[0]))

        # Explanation of result
        if result == 0:
            st.info(f"The engine is predicted to be in a normal condition. Confidence level: {1.0 - confidence:.2%}")
        else:
            st.warning(f"Warning! Please investigate further. Confidence level: {1.0 - confidence:.2%}")

    # Reset button
    if st.button("Reset Values"):
        st.script_runner.rerun()

# Function to predict engine condition
def predict_condition(engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference):
    input_data = np.array([engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference]).reshape(1, -1)
    prediction = model.predict(input_data)
    confidence = model.predict_proba(input_data)[:, 1]  # For binary classification, adjust as needed
    return prediction[0], confidence[0]

if __name__ == "__main__":
    main()
