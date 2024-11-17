import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import shap
import pickle

# Load the dataset
data = pd.read_csv('test_model.csv')

# Derive new features
data['Temperature_difference'] = data['Coolant temp'] - data['lub oil temp']
data.drop(['Engine_power'], axis=1, inplace=True, errors='ignore')

# Prepare features and target
X = data.drop(['Engine Condition'], axis=1)
y = data['Engine Condition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train GBM model
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    max_features='sqrt',
    min_samples_leaf=5,
    min_samples_split=2,
    subsample=0.8,
)
model.fit(X_train, y_train)

# Save the trained model
with open('hhmodel.pkl', 'wb') as f:
    pickle.dump(model, f)

# SHAP Explanation
explainer = shap.Explainer(model.predict_proba, X_train)

# Example input data (replace with actual values for a single entry)
single_entry = pd.DataFrame({
    'Engine rpm': [100],           # Replace with actual value
    'Lub oil pressure': [50],      # Replace with actual value
    'Fuel pressure': [120],        # Replace with actual value
    'Coolant pressure': [100],     # Replace with actual value
    'lub oil temp': [90],          # Replace with actual value
    'Coolant temp': [85],          # Replace with actual value
    'Temperature_difference': [90 - 85],  # Assuming feature already exists
})

# Align single_entry columns with X_train to ensure compatibility
single_entry = single_entry[X_train.columns]

# Make a prediction for the single entry
probability = model.predict_proba(single_entry)[0, 1]  # Probability of needing maintenance
threshold = 0.6
needs_maintenance = probability > threshold

# Maintenance status
if needs_maintenance:
    status = "Maintenance Needed"
else:
    status = "No Maintenance Needed"
print("Maintenance Status:", status)

# Get SHAP values for the single instance
shap_values = explainer(single_entry)

# Identify the most critical feature for the decision
important_feature_index = np.argmax(abs(shap_values[0].values))  # Feature with highest SHAP value
important_feature = single_entry.columns[important_feature_index]
important_feature_contribution = shap_values[0].values[important_feature_index]  # Direct access

# Display which part needs maintenance
if needs_maintenance:
    print(f"The most likely part needing maintenance is: {important_feature} "
          f"(SHAP Contribution: {important_feature_contribution:.2f})")

# SHAP Visualization (force plot for the single test instance)
shap.initjs()
force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values[0].values,
    single_entry  # Use the DataFrame directly for feature names
)

# Save the SHAP force plot to an HTML file
shap.save_html("shap_force_plot.html", force_plot)

# Debugging: Verify feature alignment and SHAP values
print("Expected features in model:", list(X_train.columns))
print("Features in single entry:", list(single_entry.columns))
print("SHAP values for the instance:", shap_values[0].values)
