import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# Set Streamlit page config and theme
st.set_page_config(layout="wide", page_title="Financial Independence Calculator")
st.markdown(
    """
    <style>
    :root {
        --primary-color: #00bcd4;
        --table-navy: #003049;
        --table-cyan: #00bcd4;
        --background-color: #ffffff;
        --secondary-background-color: #f5f7fa;
        --text-color: #111111;
        --font-family: 'sans-serif';
        --pill-border: #e0e0e0;
    }
    body, .stApp {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
        font-family: var(--font-family) !important;
    }
    .stButton>button {
        background-color: var(--primary-color);
        color: #fff;
        border-radius: 999px;
        padding: 0.5em 2em;
        font-weight: 600;
    }
    .custom-label {
        color: #111 !important;
        font-weight: 600 !important;
        font-size: 1em !important;
        margin-bottom: 0.2em !important;
        margin-left: 0.3em !important;
    }
    .custom-header-vertical {
        writing-mode: vertical-rl;
        transform: rotate(180deg);
        font-size: 1em;
        color: #003049;
        font-weight: 700;
        padding: 0.5em 0.2em;
        margin-bottom: 0.5em;
    }
    .custom-header-horizontal {
        font-size: 1em;
        color: #003049;
        font-weight: 700;
        padding: 0.5em 0.2em;
        margin-bottom: 0.5em;
    }
    .custom-pill-input input {
        background: #fff !important;
        color: #111 !important;
        border: 1.5px solid var(--pill-border) !important;
        border-radius: 999px !important;
        padding: 0.5em 1.5em !important;
        font-size: 1.1em !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        margin-bottom: 0.5em !important;
    }
    .custom-info {
        font-size: 0.9em;
        color: #888;
        margin-left: 0.3em;
        cursor: pointer;
        border-bottom: 1px dotted #888;
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


# Inputs as pill/box style above the table
st.markdown("""
<div style='padding: 1.5em 0 0.5em 0;'><h4 style='margin-bottom:0.5em;'>Current Costs and Earnings</h4></div>
""", unsafe_allow_html=True)

col_inputs = st.columns([1,1,1,1,1,1,1,1])
with col_inputs[0]:
    st.markdown("<span class='custom-label'>Current assets</span>", unsafe_allow_html=True)
    current_assets = st.number_input(" ", min_value=0, value=65000, step=1000, format="%d", key="assets", label_visibility="collapsed")
with col_inputs[1]:
    st.markdown("<span class='custom-label'>Salary p.m.</span>", unsafe_allow_html=True)
    salary_pm = st.number_input(" ", min_value=0, value=3700, step=100, format="%d", key="salary", label_visibility="collapsed")
with col_inputs[2]:
    st.markdown("<span class='custom-label'>Expenses p.m.</span>", unsafe_allow_html=True)
    expenses = st.number_input(" ", min_value=0, value=2000, step=100, format="%d", key="expenses", label_visibility="collapsed")
with col_inputs[3]:
    st.markdown("<span class='custom-label'>Market p.a. (%)</span>", unsafe_allow_html=True)
    market_pa = st.number_input(" ", min_value=0.0, value=4.0, step=0.1, format="%.2f", key="market", label_visibility="collapsed")
with col_inputs[4]:
    st.markdown("<span class='custom-label'>Salary Tax (%) <span class='custom-info' title='Total of social security contributions and taxes as a % of gross salary.'>i</span></span>", unsafe_allow_html=True)
    salary_tax = st.number_input(" ", min_value=0.0, value=35.0, step=0.1, format="%.1f", key="salarytax", label_visibility="collapsed")
with col_inputs[5]:
    st.markdown("<span class='custom-label'>Capital Tax (%)</span>", unsafe_allow_html=True)
    capital_tax = st.number_input(" ", min_value=0.0, value=25.0, step=0.1, format="%.1f", key="captax", label_visibility="collapsed")
with col_inputs[6]:
    st.markdown("<span class='custom-label'>Invested (%) <span class='custom-info' title='Share of your total money that is not held in cash.'>i</span></span>", unsafe_allow_html=True)
    invested = st.number_input(" ", min_value=0.0, max_value=100.0, value=100.0, step=1.0, format="%.1f", key="invested", label_visibility="collapsed")
with col_inputs[7]:
    st.markdown("<span class='custom-label'>Your Age</span>", unsafe_allow_html=True)
    current_age = st.number_input(" ", min_value=0, value=33, step=1, format="%d", key="age", label_visibility="collapsed")


# --- Calculation Logic ---
import numpy as np


# User Inputs
gross_income_annual = salary_pm * 12
net_income_annual = gross_income_annual * (1 - salary_tax / 100)
annual_expenses = expenses * 12  # current (pre-FI) annual expenses
invested_fraction = invested / 100
market_return = market_pa / 100
capital_tax_fraction = capital_tax / 100

# Table axes
base_salary = salary_pm * 12
raw_salary_cols = [int(round(base_salary/10000)*10000)] + [int(round(base_salary * (1 + 0.2 * i)/10000)*10000) for i in range(1, 7)]
# Ensure unique column names for display, keep numeric values for calculation
salary_labels = []
seen = {}
for s in raw_salary_cols:
    label = f"€{int(s):,}"
    if label in seen:
        seen[label] += 1
        label = f"{label} ({seen[label]})"
    else:
        seen[label] = 1
    salary_labels.append(label)

# Ensure salary_labels are unique for DataFrame columns
from collections import Counter
label_counts = Counter(salary_labels)
unique_labels = []
label_tracker = {}
for label in salary_labels:
    if label_counts[label] > 1:
        if label not in label_tracker:
            label_tracker[label] = 1
        else:
            label_tracker[label] += 1
        unique_labels.append(f"{label}_{label_tracker[label]}")
    else:
        unique_labels.append(label)
salary_labels = unique_labels


# Target Expenses Needed per Month (pill style)


# Target Expenses Needed per Month (pill style)
st.markdown("<span class='custom-label'>Target Monthly Expenses <span class='custom-info' title='Adjust your target monthly expenses to see how it affects your FI age.'>i</span></span>", unsafe_allow_html=True)
target_spending = st.number_input(" ", min_value=0, value=2000, step=100, format="%d", key="targetspend_main", label_visibility="collapsed")

# Target spending: center row is input, increments of 500 up/down, floor 500, remove bottom 5 rows
spending_center = target_spending
spending_rows = [max(500, spending_center - 2500),
                 max(500, spending_center - 2000),
                 max(500, spending_center - 1500),
                 max(500, spending_center - 1000),
                 max(500, spending_center - 500),
                 spending_center,
                 spending_center + 500,
                 spending_center + 1000,
                 spending_center + 1500,
                 spending_center + 2000,
                 spending_center + 2500]
spending_rows = [x for x in spending_rows if x >= 500]
spending_rows = spending_rows[:6] + spending_rows[6:7]  # Only keep center and 5 above

# Function to simulate wealth trajectory and FI age
def calculate_fi_age(
    current_assets,
    net_income_annual,
    annual_current_expenses,
    market_return,
    capital_tax_fraction,
    invested_fraction,
    current_age,
    target_spending,
):
    assets = current_assets
    age = current_age
    while age < 100:
        # Annual savings = net salary minus current living expenses
        savings = net_income_annual - annual_current_expenses
        # Investment returns (after tax) only on invested portion of assets
        investment_return = assets * invested_fraction * market_return * (1 - capital_tax_fraction)
        # Update assets with savings and investment return
        assets = assets + savings + investment_return
        # Check if investment return covers target spending
        if investment_return / 12 >= target_spending:
            return age
        age += 1
    return None

# Build results table
results = []
for spend in spending_rows:
    row = []
    for income in raw_salary_cols:
        net_income = income * (1 - salary_tax / 100)
        fi_age = calculate_fi_age(
            current_assets=current_assets,
            net_income_annual=net_income,
            annual_current_expenses=annual_expenses,
            market_return=market_return,
            capital_tax_fraction=capital_tax_fraction,
            invested_fraction=invested_fraction,
            current_age=current_age,
            target_spending=spend,
        )
        row.append(fi_age if fi_age is not None else np.nan)
    results.append(row)

results_df = pd.DataFrame(
    results,
    index=[f"€{int(x):,}" for x in spending_rows],
    columns=salary_labels
)


# Table with blue-only color palette and clarified headers

st.subheader("Age when Capital Returns Cover Annual Expenses")
styled_df = results_df.copy()
styled_df.index.name = "Expenses need p.m. (€)"
styled_df.columns.name = "Annual Salary (€)"

# Layout: horizontal header above
st.markdown("<div class='custom-header-horizontal'>Annual Salary (€)</div>", unsafe_allow_html=True)

# Heatmap with blue gradient to visualize FI age
fig, ax = plt.subplots(figsize=(8, 4))
data = styled_df.values.astype(float)
im = ax.imshow(data, aspect="auto", cmap="Blues", origin="upper")
ax.set_xticks(range(len(styled_df.columns)))
ax.set_xticklabels(styled_df.columns, rotation=45, ha="right")
ax.set_yticks(range(len(styled_df.index)))
ax.set_yticklabels(styled_df.index)
ax.set_xlabel("Annual Salary (€)")
ax.set_ylabel("Expenses need p.m. (€)")

# Write FI age inside each cell
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{int(val)}", ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=ax, label="FI Age")
st.pyplot(fig)

# Calculate summary for selected target spending
def get_summary(
    current_assets,
    net_income_annual,
    annual_current_expenses,
    market_return,
    capital_tax_fraction,
    invested_fraction,
    current_age,
    target_spending,
):
    assets = current_assets
    age = current_age
    while age < 100:
        savings = net_income_annual - annual_current_expenses
        investment_return = assets * invested_fraction * market_return * (1 - capital_tax_fraction)
        assets = assets + savings + investment_return
        if investment_return / 12 >= target_spending:
            return age, investment_return / 12
        age += 1
    return None, investment_return / 12

fi_age, monthly_return = get_summary(
    current_assets=current_assets,
    net_income_annual=net_income_annual,
    annual_current_expenses=annual_expenses,
    market_return=market_return,
    capital_tax_fraction=capital_tax_fraction,
    invested_fraction=invested_fraction,
    current_age=current_age,
    target_spending=target_spending,
)

if fi_age:
    summary_text = f"You can expect to reach financial independence at age <b>{fi_age}</b>. Your investments could generate approximately <b>{monthly_return:,.0f} EUR</b> per month at that point."
else:
    summary_text = f"You won't be able to cover your target expenses budget without working. You can expect <b>{monthly_return:,.0f} EUR</b> per month from your investments by age 100."

st.markdown(
    f"""
    <div style='background-color:#003049; color:#fff; padding:20px; border-radius:10px; margin-top:30px;'>
        <h4>Financial Independence Break Even</h4>
        <p>{summary_text}</p>
    </div>
    """,
    unsafe_allow_html=True
)


# --- Net Worth Over Time Chart ---

def simulate_net_worth(
    current_assets,
    net_income_annual,
    annual_current_expenses,
    market_return,
    capital_tax_fraction,
    invested_fraction,
    current_age,
):
    """Simulate yearly net worth from current age to 100 using same assumptions."""
    assets = current_assets
    ages = []
    values = []
    age = current_age
    while age <= 100:
        ages.append(age)
        values.append(assets)
        savings = net_income_annual - annual_current_expenses
        investment_return = assets * invested_fraction * market_return * (1 - capital_tax_fraction)
        assets = assets + savings + investment_return
        age += 1
    return ages, values


ages, net_worth = simulate_net_worth(
    current_assets=current_assets,
    net_income_annual=net_income_annual,
    annual_current_expenses=annual_expenses,
    market_return=market_return,
    capital_tax_fraction=capital_tax_fraction,
    invested_fraction=invested_fraction,
    current_age=current_age,
)

# Compute target FI capital level for reference (if feasible)
target_capital = None
denominator = invested_fraction * market_return * (1 - capital_tax_fraction)
if denominator > 0 and target_spending > 0:
    target_capital = (target_spending * 12) / denominator

st.markdown("<div style='height: 32px'></div>", unsafe_allow_html=True)
st.subheader("Projected Net Worth Over Time")
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(ages, net_worth, color="#003049", linewidth=2, label="Net worth")

if target_capital is not None:
    ax2.axhline(target_capital, color="#00bcd4", linestyle="--", linewidth=1.5, label="FI target capital")

ax2.set_xlabel("Age (years)")
ax2.set_ylabel("Net worth (€)")
ax2.ticklabel_format(style="plain", axis="y")
ax2.grid(alpha=0.15)
ax2.legend()
st.pyplot(fig2)
