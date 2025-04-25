import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Load your coefficients
coeffs = pd.read_csv('state_medicaid_model_coeffs.csv')

# Set up the app
st.title("Medicaid Eligibility Change Predictor")
st.write("Predict how changes in employment rates affect Medicaid eligibility by state.")

# User picks a state
state = st.selectbox("Select a state:", coeffs['State'].tolist())

# Get coefficients for that state
state_row = coeffs[coeffs['State'] == state]
employment_coeff = state_row['EmploymentRateCoeff'].values[0]
intercept = state_row['Intercept'].values[0]

# Employment rate change input - SLIDER
st.subheader("Change in Employment Rate")
change_in_employment = st.slider(
    "Slide to select employment rate change (%):",
    min_value=-10.0,
    max_value=10.0,
    value=0.0,
    step=0.1,
    format="%.1f"
)

# Convert percentage back to decimal
change_in_employment_decimal = change_in_employment / 100

# Calculate predicted change
predicted_change = employment_coeff * change_in_employment_decimal + intercept

# Show result
st.metric(
    label=f"Predicted Change in Medicaid-Eligible Population for {state}",
    value=f"{predicted_change*100:.2f}%",
    delta=f"{predicted_change*100 - intercept*100:.2f}%"  # Show how much changed compared to baseline
)

# Bonus: plot a little chart
st.subheader("Sensitivity Analysis: Employment Rate Change vs Medicaid Eligibility")

x = np.linspace(-10, 10, 100)  # Employment rate changes from -10% to +10%
y = employment_coeff * (x / 100) + intercept

fig, ax = plt.subplots()
ax.plot(x, y * 100)  # Convert to percentages
ax.axvline(change_in_employment, color='red', linestyle='--')  # Current user input
ax.set_xlabel('Change in Employment Rate (%)')
ax.set_ylabel('Change in Medicaid Eligible Share (%)')
ax.set_title(f"Predicted Impact for {state}")
ax.grid(True)

st.pyplot(fig)
