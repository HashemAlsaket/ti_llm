import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from openai import OpenAI  # Updated import for new OpenAI SDK

# --- SETUP ---
st.set_page_config(page_title="TI LLM Agent", layout="wide")

# --- LOGIN FLOW ---
def login():
    st.session_state["authenticated"] = False

    with st.form("Login"):
        st.write("🔐 Please log in to continue")
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if user == st.secrets["credentials"]["username"] and pw == st.secrets["credentials"]["password"]:
                st.session_state["authenticated"] = True
            else:
                st.error("Invalid credentials")

if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    login()
    st.stop()

st.title("📊 TI LLM Agent Prototype")

# --- INITIALIZE OPENAI CLIENT ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # Updated client setup

# --- MOCK DATA ---
def load_mock_data():
    np.random.seed(42)
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA']
    model_groups = ['Macro Alpha', 'Q1 Equity', 'Commodities Signal', 'Rates Momentum']

    data = [
        {
            "ticker": np.random.choice(tickers),
            "model_group": np.random.choice(model_groups),
            "timestamp": datetime(2024, 4, np.random.randint(1, 30)),
            "position": np.random.uniform(-1000000, 1000000),
            "pnl": np.random.uniform(-100000, 100000),
            "alpha_score": np.random.normal(0, 1),
            "volatility": np.random.uniform(0.1, 0.5)
        }
        for _ in range(500)
    ]
    return pd.DataFrame(data)

# --- LLM QUERY FUNCTION ---
def ask_llm(prompt, context):
    system_prompt = (
        "You are a financial analyst assistant for Tudor Investments. "
        "Answer questions using the following data context."
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Data context: {context}\n\nUser question: {prompt}"},
        ],
        temperature=0.2,
        max_tokens=500
    )
    return response.choices[0].message.content

# --- LOAD DATA ---
df = load_mock_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔍 Filter Trades")
ticker = st.sidebar.selectbox("Select Ticker", ["All"] + sorted(df['ticker'].unique().tolist()))
model = st.sidebar.selectbox("Select Model Group", ["All"] + sorted(df['model_group'].unique().tolist()))

filtered_df = df.copy()
if ticker != "All":
    filtered_df = filtered_df[filtered_df['ticker'] == ticker]
if model != "All":
    filtered_df = filtered_df[filtered_df['model_group'] == model]

# --- DISPLAY DATA ---
st.subheader("🔁 Trade Data")
st.dataframe(filtered_df, use_container_width=True)

# --- SUMMARY METRICS ---
total_pnl = filtered_df['pnl'].sum()
total_position = filtered_df['position'].sum()
avg_alpha = filtered_df['alpha_score'].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Total PnL", f"${total_pnl:,.0f}")
col2.metric("Net Position", f"${total_position:,.0f}")
col3.metric("Avg Alpha Score", f"{avg_alpha:.2f}")

# --- LLM SIMULATION ---
st.subheader("🤖 Ask a Question")
user_prompt = st.text_area("What would you like to know about this trading data?")

if st.button("Ask the LLM"):
    if user_prompt:
        # Use a shortened context for proof of concept
        sample_data = filtered_df.head(10).to_dict(orient="records")
        context_text = str(sample_data)

        with st.spinner("Asking the LLM..."):
            answer = ask_llm(user_prompt, context_text)
        st.success("LLM Response:")
        st.write(answer)
    else:
        st.warning("Please enter a question.")