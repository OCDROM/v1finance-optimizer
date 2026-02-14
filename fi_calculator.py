import streamlit as st
import pandas as pd

# Set Streamlit page config and theme
st.set_page_config(layout="wide", page_title="Financial Independence Calculator")
st.markdown(
    """
    <style>
    :root {
        --primary-color: #00bcd4;
        --background-color: #003049;
        --secondary-background-color: #011f30;
        --text-color: #FFFFFF;
        --font-family: 'sans-serif';
    }
    body, .stApp {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
        font-family: var(--font-family) !important;
    }
    .stButton>button {
        background-color: var(--primary-color);
        color: var(--text-color);
    }
    .stTextInput>div>input {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
    }
    .stNumberInput>div>input {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
    }
    .stDataFrame, .stTable {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Financial Independence Calculator")

# Sidebar for user inputs
with st.sidebar:
    st.header("Current costs and earnings")
    current_assets = st.number_input("Current assets", min_value=0, value=1173891, step=1000, format="%d")
    salary_pm = st.number_input("Salary p.m.", min_value=0, value=14167, step=100, format="%d")
    expenses = st.number_input("Expenses", min_value=0, value=3000, step=100, format="%d")
    market_pa = st.number_input("Market p.a. (%)", min_value=0.0, value=3.5, step=0.1, format="%.2f")
    bonus = st.number_input("Bonus (months)", min_value=0.0, value=4.5, step=0.1, format="%.1f")
    salary_tax = st.number_input("Salary Tax (%)", min_value=0.0, value=35.0, step=0.1, format="%.1f")
    capital_tax = st.number_input("Capital Tax (%)", min_value=0.0, value=25.0, step=0.1, format="%.1f")
    invested = st.number_input("Invested (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0, format="%.1f")
    current_age = st.number_input("Your Current Age", min_value=0, value=33, step=1, format="%d")


# --- Calculation Logic ---
import numpy as np

# User Inputs
gross_income_annual = salary_pm * 12 + bonus * salary_pm
net_income_annual = gross_income_annual * (1 - salary_tax / 100)
annual_expenses = expenses * 12
invested_fraction = invested / 100
market_return = market_pa / 100
capital_tax_fraction = capital_tax / 100

# Table axes
target_spending_range = np.arange(1000, 7001, 500)  # €1,000 to €7,000
gross_income_range = np.arange(80000, 380001, 50000)  # €80,000 to €380,000

# Function to simulate wealth trajectory and FI age
def calculate_fi_age(
    current_assets, net_income_annual, annual_expenses, market_return, capital_tax_fraction, invested_fraction, current_age, target_spending
):
    assets = current_assets
    age = current_age
    while age < 100:
        # Investment returns (after tax)
        investment_return = assets * market_return * (1 - capital_tax_fraction)
        # Add net income and investment return, subtract expenses
        assets = assets + net_income_annual * invested_fraction + investment_return - annual_expenses
        # Check if investment return covers target spending
        if investment_return / 12 >= target_spending:
            return age
        age += 1
    return None

# Build results table
results = []
for spend in target_spending_range:
    row = []
    for income in gross_income_range:
        net_income = income * (1 - salary_tax / 100)
        fi_age = calculate_fi_age(
            current_assets=current_assets,
            net_income_annual=net_income,
            annual_expenses=spend * 12,
            market_return=market_return,
            capital_tax_fraction=capital_tax_fraction,
            invested_fraction=invested_fraction,
            current_age=current_age,
            target_spending=spend
        )
        row.append(fi_age if fi_age is not None else np.nan)
    results.append(row)

results_df = pd.DataFrame(
    results,
    index=[f"€{int(x):,}" for x in target_spending_range],
    columns=[f"€{int(x):,}" for x in gross_income_range]
)

st.subheader("Target Monthly Spending after Financial Independence")
target_spending = st.number_input("Target Monthly Spending", min_value=0, value=3000, step=100, format="%d")
st.write(":blue[Adjust your target monthly spending to see how it affects your FI age]")

st.subheader("Age when Capital Returns Cover Annual Expenses")
st.dataframe(results_df.style.background_gradient(cmap="PuBuGn"), height=400)

# Calculate summary for selected target spending
def get_summary(current_assets, net_income_annual, annual_expenses, market_return, capital_tax_fraction, invested_fraction, current_age, target_spending):
    assets = current_assets
    age = current_age
    while age < 100:
        investment_return = assets * market_return * (1 - capital_tax_fraction)
        assets = assets + net_income_annual * invested_fraction + investment_return - annual_expenses
        if investment_return / 12 >= target_spending:
            return age, investment_return / 12
        age += 1
    return None, investment_return / 12

fi_age, monthly_return = get_summary(
    current_assets=current_assets,
    net_income_annual=net_income_annual,
    annual_expenses=target_spending * 12,
    market_return=market_return,
    capital_tax_fraction=capital_tax_fraction,
    invested_fraction=invested_fraction,
    current_age=current_age,
    target_spending=target_spending
)

if fi_age:
    summary_text = f"You can expect to reach financial independence at age <b>{fi_age}</b>. Your investments could generate approximately <b>{monthly_return:,.0f} EUR</b> per month at that point."
else:
    summary_text = f"You won't be able to cover your target expenses budget without working. You can expect <b>{monthly_return:,.0f} EUR</b> per month from your investments by age 100."

st.markdown(
    f"""
    <div style='background-color:#011f30; color:#00bcd4; padding:20px; border-radius:10px; margin-top:30px;'>
        <h4>Financial Independence Cross Over</h4>
        <p>{summary_text}</p>
    </div>
    """,
    unsafe_allow_html=True
)
