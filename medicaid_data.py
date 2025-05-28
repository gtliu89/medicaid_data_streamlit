import pandas as pd
import numpy as np
import streamlit as st
st.set_page_config(page_title="Medicaid Eligibility Prediction", layout="wide")
import plotly.express as px
import pydeck as pdk

state_abbr = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI',
    'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND',
    'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY'
}

# Load and cache data
@st.cache_data
def load_data():
    coeffs = pd.read_csv("model_output_2025.05.06.csv")
    poverty_inputs = pd.read_csv("2023_dataset_ests_2025.05.07.csv")
    return coeffs, poverty_inputs

coeffs, poverty_inputs = load_data()

# Extract valid states
state_rows = coeffs[coeffs["variable"].str.startswith("state_")]
available_states = state_rows["variable"].str.replace("state_", "", regex=False).tolist()

# Mapping helpers
group_col_map = {"adults": "adult", "kids": "child"}

# Predict function
def predict_change(group: str, state_var: str, emp_change_decimal: float) -> float:
    group_col = group_col_map[group]
    
    intercept = coeffs.loc[coeffs["variable"] == "Intercept", group_col].values[0]
    employment = coeffs.loc[coeffs["variable"] == "L.ln_employ", group_col].values[0]
    state_effect = coeffs.loc[coeffs["variable"] == state_var, group_col].values[0]
    prediction = intercept + (employment * emp_change_decimal) + state_effect

    poverty_row = poverty_inputs[poverty_inputs["state"] == state_var]
    if poverty_row.empty:
        return np.nan

    for var in coeffs["variable"]:
        if var.startswith("L.povcat11_"):
            coeff_val = coeffs.loc[coeffs["variable"] == var, group_col].values[0]
            cat = var.split("_")[-1]
            input_col = f"povcat11_{group}_{cat}_est2"
            if input_col in poverty_row.columns:
                prediction += (coeff_val * poverty_row[input_col].values[0])

    return prediction


# Page UI
st.title("📊 Medicaid Eligibility Prediction Tool")
st.markdown("Select a state and input a change in employment to see how it would impact Medicaid eligibility.")

state_choice = st.selectbox("Select a state", sorted(available_states), index=available_states.index("Minnesota"))
state_var = f"state_{state_choice}"

current_employment = poverty_inputs.loc[poverty_inputs['state'] == state_var, 'employ_est2'].values[0]
emprate = current_employment*100

st.write(f"The current employment rate in {state_choice} is **{current_employment * 100:.2f}%**")

emp_change = st.slider("Drag the slider to select a level of employment and see how it effects Medicaid eligibility.", emprate - 10.0, emprate + 10.0, step=0.1, value=emprate)

# Calculate predictions
emp_change_decimal = ((emp_change - emprate)/emprate) * 100

adults_result = predict_change("adults", state_var, emp_change_decimal)
kids_result = predict_change("kids", state_var, emp_change_decimal)

adult_basecount = poverty_inputs.loc[poverty_inputs['state'] == state_var, 'elig_all_adult_w2'].values[0]
kids_basecount = poverty_inputs.loc[poverty_inputs['state'] == state_var, 'elig_all_child_w2'].values[0]
total_basecount = adult_basecount + kids_basecount
total_eligible = (adult_basecount * (adults_result/100 + 1)) + (kids_basecount * (kids_result/100 + 1))
delta = total_eligible - total_basecount

st.write(f"You have entered a change in employment of {emp_change - emprate:.2f}%.")

# Display Metrics
st.markdown("### Number of individuals eligible for Medicaid")
m1, m2, m3 = st.columns(3)
m1.metric("Aduluts eligible", f"{adult_basecount * (adults_result/100 + 1):,.0f}")
m2.metric("Kids eligible", f"{kids_basecount * (kids_result/100 + 1):,.0f}")
m3.metric("Total eligible", f"{total_eligible:,.0f}", f"{delta:+,.0f}")

st.markdown("### Predicted Percent Change in Medicaid Eligibility Rate")
m1, m2, m3 = st.columns(3)
m1.metric("👩 Change in Adult Eligibility", f"{adults_result:.2f} %")
m2.metric("🧒 Change in Child Eligibility", f"{kids_result:.2f} %")
m3.metric("Total change in Eligibility", f"{((delta)/total_basecount)*100:.2f} %")



# Line chart over employment range
st.markdown("### 📈 Eligibility Over Employment Rate Changes")
x_vals = np.linspace(-10, 10, 100)
x_vals_adjusted = [((emprate + change) - emprate) / emprate * 100 for change in x_vals]
adults_preds = [predict_change("adults", state_var, x) for x in x_vals_adjusted]
kids_preds = [predict_change("kids", state_var, x) for x in x_vals_adjusted]

chart_df = pd.DataFrame({
    "Employment Change (%)": x_vals,
    "Adults": adults_preds,
    "Children": kids_preds
})

fig = px.line(chart_df, x="Employment Change (%)", y=["Adults", "Children"],
              labels={"value": "Eligibility %"}, title=f"Effect of Employment Rate Change on Eligibility in {state_choice}")
fig.update_traces(hovertemplate="%{y:.2f}% change in eligibility at %{x:.2f}% employment change")

fig.update_yaxes(range=[-20, 20])
fig.update_xaxes(range=[-10, 10])

st.plotly_chart(fig, use_container_width=True)
