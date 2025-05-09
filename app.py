import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_community.utilities import SQLDatabase

# --- SETUP ---
st.set_page_config(page_title="TI LLM Agent", layout="wide")

# --- DATABASE SETUP ---
DB_PATH = "finance_data.db"

def init_db():
    """Initialize SQLite database and create tables if they don't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create trades table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        model_group TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        position REAL NOT NULL,
        pnl REAL NOT NULL,
        alpha_score REAL NOT NULL,
        volatility REAL NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()

def db_is_empty():
    """Check if database is empty"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM trades")
    count = cursor.fetchone()[0]
    conn.close()
    return count == 0

def load_data_to_db():
    """Generate mock data and load into SQLite database"""
    np.random.seed(42)
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA']
    model_groups = ['Macro Alpha', 'Q1 Equity', 'Commodities Signal', 'Rates Momentum']

    # Generate mock data
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
    df = pd.DataFrame(data)
    
    # Insert data into database
    conn = sqlite3.connect(DB_PATH)
    df.to_sql('trades', conn, if_exists='append', index=False)
    conn.commit()
    conn.close()
    
    return df

def get_data_from_db(ticker="All", model_group="All"):
    """Fetch data from SQLite database with optional filters"""
    conn = sqlite3.connect(DB_PATH)
    
    query = "SELECT * FROM trades"
    params = []
    
    # Add filters if specified
    where_clauses = []
    if ticker != "All":
        where_clauses.append("ticker = ?")
        params.append(ticker)
    if model_group != "All":
        where_clauses.append("model_group = ?")
        params.append(model_group)
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df

def get_detailed_db_schema():
    """Get a detailed database schema description for better LLM context"""
    schema = """
Table: trades
Description: This table contains financial trading data for various stock tickers across different trading models.

Columns:
* id (INTEGER) # Unique identifier for each trade record
* ticker (TEXT) # Stock ticker symbol (e.g., AAPL, MSFT, GOOG) representing the company being traded
* model_group (TEXT) # The trading strategy or model used for the trade (e.g., Macro Alpha, Q1 Equity)
* timestamp (TIMESTAMP) # Date and time when the trade was executed or recorded
* position (REAL) # Current position size in dollars; positive values indicate long positions, negative values indicate short positions
* pnl (REAL) # Profit and Loss in dollars; indicates how much profit (positive) or loss (negative) the position has generated
* alpha_score (REAL) # A score measuring the excess return of the investment relative to a benchmark; higher values indicate better performance
* volatility (REAL) # A measure of the price variation/risk of the position; higher values indicate more volatile/risky positions
"""
    return schema

def get_db_schema():
    """Get the database schema as a string"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema = []
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        col_defs = [f"{col[1]} {col[2]}" for col in columns]
        schema.append(f"Table: {table_name}\nColumns: {', '.join(col_defs)}")
    
    conn.close()
    return "\n\n".join(schema)

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

# Add simple Tudor branding
st.markdown("""
<style>
    .main-header {color: #0F4B81; font-size: 26px; font-weight: bold;}
    .section-header {color: #0F4B81; font-size: 20px;}
    .stButton>button {background-color: #0F4B81; color: white;}
</style>
""", unsafe_allow_html=True)

st.markdown("<p class='main-header'>📊 Tudor Investments LLM Agent</p>", unsafe_allow_html=True)

# --- INITIALIZE DATABASE ---
init_db()

# Load sample data if database is empty
if db_is_empty():
    with st.spinner("Loading initial data..."):
        load_data_to_db()
        st.success("Sample data loaded successfully!")

# --- INITIALIZE CLIENTS ---
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
langchain_llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4")

# --- SETUP LANGCHAIN SQL DATABASE ---
db_url = f"sqlite:///{DB_PATH}"
db = SQLDatabase.from_uri(db_url)

# --- LLM QUERY FUNCTIONS ---
def ask_direct_llm(prompt, context):
    """Legacy direct LLM query method"""
    # Include detailed schema for better context
    detailed_schema = get_detailed_db_schema()
    
    system_prompt = (
        "You are a financial analyst assistant for Tudor Investments. "
        "Answer questions using the following database schema and data context.\n\n"
        f"Database Schema:\n{detailed_schema}\n\n"
        "Answer questions using the following data context."
    )

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Data context: {context}\n\nUser question: {prompt}"},
        ],
        temperature=0.2,
        max_tokens=500
    )
    return response.choices[0].message.content

def ask_sql_llm(prompt):
    """Use LangChain to convert natural language to SQL and execute"""
    # Get detailed schema for better SQL generation
    schema = get_detailed_db_schema()
    
    # Define an enhanced SQL generation prompt that includes the detailed schema
    sql_generation_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a financial database expert at Tudor Investments. "
            "Your job is to convert user questions into correct SQL queries based on the database schema below.\n\n"
            f"{schema}\n\n"
            "Generate only the SQL query without any explanation. The query should be syntactically correct for SQLite."
            "Think step by step about how to join tables if needed, and ensure all column names are correct."
        )),
        ("human", "{question}")
    ])
    
    # Create a chain that generates SQL
    sql_chain = create_sql_query_chain(
        langchain_llm,
        db,
        prompt=sql_generation_prompt,
        k=3  # Number of examples used for few-shot prompting
    )
    
    try:
        # Generate SQL query
        sql_query = sql_chain.invoke({"question": prompt})
        
        # Execute the query
        conn = sqlite3.connect(DB_PATH)
        results = pd.read_sql(sql_query, conn)
        conn.close()
        
        # Get the LLM to explain the results
        explain_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "You are a financial analyst assistant for Tudor Investments. "
                "Explain the following SQL query and its results in a clear, concise way. "
                "Focus on the business insights and implications.\n\n"
                f"Database Schema:\n{schema}"
            )),
            ("human", "SQL Query: {query}\n\nResults: {results}\n\nUser question: {question}")
        ])
        
        chain = explain_prompt | langchain_llm | StrOutputParser()
        explanation = chain.invoke({
            "query": sql_query,
            "results": results.to_string(),
            "question": prompt
        })
        
        return {
            "sql": sql_query,
            "results": results,
            "explanation": explanation
        }
    except Exception as e:
        return {
            "error": str(e),
            "sql": "Error generating or executing SQL"
        }

def ask_enhanced_sql_llm(prompt, filtered_df):
    """Enhanced SQL-based approach with better context"""
    # Get detailed schema
    schema = get_detailed_db_schema()
    
    # Try SQL approach
    sql_response = ask_sql_llm(prompt)
    
    # Create a combined prompt with SQL results and filtered data context
    combined_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a sophisticated financial analyst assistant for Tudor Investments. "
            "Provide a comprehensive answer to the user's question using the SQL query results, "
            "the current filtered data context, and the database schema.\n\n"
            f"Database Schema:\n{schema}"
        )),
        ("human", """
        User question: {question}
        
        SQL Analysis:
        {sql_results}
        
        Current Filtered Data Summary:
        {filtered_data_summary}
        
        Please provide a comprehensive answer that addresses the user's question directly.
        Include specific numbers and insights from the data where relevant.
        If appropriate, suggest follow-up analyses that might provide additional insights.
        """)
    ])
    
    # Create a summary of the filtered data
    filtered_summary = f"Current filter: Ticker={ticker}, Model Group={model}"
    if ticker != "All" or model != "All":
        filtered_summary += f"\nFiltered dataset contains {len(filtered_df)} records."
        filtered_summary += f"\nSummary statistics for filtered data:"
        filtered_summary += f"\n- Total PnL: ${filtered_df['pnl'].sum():,.2f}"
        filtered_summary += f"\n- Average position size: ${filtered_df['position'].mean():,.2f}"
        filtered_summary += f"\n- Average alpha score: {filtered_df['alpha_score'].mean():.4f}"
    
    if "error" in sql_response:
        sql_results_text = f"SQL Error: {sql_response['error']}"
    else:
        sql_results_text = f"SQL Query: {sql_response['sql']}\n\nResults Summary: {sql_response.get('explanation', 'No explanation available')}"
    
    chain = combined_prompt | langchain_llm | StrOutputParser()
    final_response = chain.invoke({
        "question": prompt,
        "sql_results": sql_results_text,
        "filtered_data_summary": filtered_summary
    })
    
    return {
        "response": final_response,
        "sql": sql_response.get("sql", "No SQL query generated"),
        "sql_results": sql_response.get("results", pd.DataFrame()) if "error" not in sql_response else None
    }

# --- SIDEBAR FILTERS ---
st.sidebar.markdown("<p class='section-header'>🔍 Filter Trades</p>", unsafe_allow_html=True)
ticker_options = ["All"]
model_options = ["All"]

# Get unique values from the database
conn = sqlite3.connect(DB_PATH)
ticker_options += [row[0] for row in conn.execute("SELECT DISTINCT ticker FROM trades ORDER BY ticker").fetchall()]
model_options += [row[0] for row in conn.execute("SELECT DISTINCT model_group FROM trades ORDER BY model_group").fetchall()]
conn.close()

ticker = st.sidebar.selectbox("Select Ticker", ticker_options)
model = st.sidebar.selectbox("Select Model Group", model_options)

# --- QUERY APPROACH SELECTION ---
st.sidebar.markdown("<p class='section-header'>💡 Query Method</p>", unsafe_allow_html=True)
query_method = st.sidebar.radio(
    "LLM Query Method",
    ["Simple (Direct)", "SQL-Based", "Enhanced SQL"],
    help="Choose how the LLM will process your query"
)

# --- SAMPLE QUESTIONS ---
st.sidebar.markdown("<p class='section-header'>❓ Sample Questions</p>", unsafe_allow_html=True)
sample_questions = [
    "Which model group has the highest total PnL?",
    "What is the relationship between alpha score and volatility?",
    "Compare the average position size across different tickers",
    "Which ticker has the most negative position?",
    "Show me the trading performance by model group",
    "Calculate the correlation between position size and PnL",
    "Which model has the highest average alpha score?",
    "What is the distribution of volatility values across different tickers?",
    "Compare the PnL performance of AAPL vs MSFT",
    "Show me the model groups with positive average PnL"
]
selected_question = st.sidebar.selectbox("Try a sample question:", [""] + sample_questions)

# --- GET FILTERED DATA ---
filtered_df = get_data_from_db(ticker, model)

# --- DISPLAY DATA ---
st.markdown("<p class='section-header'>🔁 Trade Data</p>", unsafe_allow_html=True)
st.dataframe(filtered_df, use_container_width=True)

# --- SUMMARY METRICS ---
total_pnl = filtered_df['pnl'].sum()
total_position = filtered_df['position'].sum()
avg_alpha = filtered_df['alpha_score'].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Total PnL", f"${total_pnl:,.0f}")
col2.metric("Net Position", f"${total_position:,.0f}")
col3.metric("Avg Alpha Score", f"{avg_alpha:.2f}")

# --- SCHEMA DISPLAY ---
with st.expander("View Database Schema"):
    st.code(get_detailed_db_schema(), language="markdown")

# --- LLM QUERY INTERFACE ---
st.markdown("<p class='section-header'>🤖 Ask a Question</p>", unsafe_allow_html=True)

# Use the selected sample question if one is chosen
if selected_question:
    user_prompt = st.text_area("What would you like to know about this trading data?", 
                             value=selected_question,
                             placeholder="Example: Which model group has the highest PnL?")
else:
    user_prompt = st.text_area("What would you like to know about this trading data?", 
                             placeholder="Example: Which model group has the highest PnL?")

if st.button("Ask the Tudor LLM Agent"):
    if user_prompt:
        with st.spinner("Processing your question..."):
            if query_method == "Simple (Direct)":
                # Use the original direct approach
                sample_data = filtered_df.head(10).to_dict(orient="records")
                # Convert to string with better formatting to avoid JSON errors
                context_text = str(sample_data).replace("'", '"')
                answer = ask_direct_llm(user_prompt, context_text)
                
                st.success("LLM Response:")
                st.write(answer)
                
            elif query_method == "SQL-Based":
                # Use the SQL-based approach
                result = ask_sql_llm(user_prompt)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success("LLM Response:")
                    st.write(result["explanation"])
                    
                    with st.expander("View SQL Query"):
                        st.code(result["sql"], language="sql")
                    
                    with st.expander("View Raw Results"):
                        st.dataframe(result["results"])
                
            else:  # Enhanced SQL
                # Use the enhanced SQL approach
                result = ask_enhanced_sql_llm(user_prompt, filtered_df)
                
                st.success("LLM Response:")
                st.write(result["response"])
                
                with st.expander("View Technical Details"):
                    st.subheader("SQL Query")
                    st.code(result["sql"], language="sql")
                    
                    if result["sql_results"] is not None:
                        st.subheader("SQL Results")
                        st.dataframe(result["sql_results"])
                    else:
                        st.warning("SQL query execution failed")
    else:
        st.warning("Please enter a question.")

# --- ADD DATA UPLOAD FUNCTIONALITY ---
st.markdown("<p class='section-header'>📤 Upload Additional Data</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CSV file with trading data", type=["csv"])
if uploaded_file is not None:
    # Read uploaded CSV
    try:
        upload_df = pd.read_csv(uploaded_file)
        required_columns = ["ticker", "model_group", "timestamp", "position", "pnl", "alpha_score", "volatility"]
        
        # Check if required columns exist
        missing_cols = [col for col in required_columns if col not in upload_df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
        else:
            # Convert timestamp if needed
            if not pd.api.types.is_datetime64_any_dtype(upload_df["timestamp"]):
                upload_df["timestamp"] = pd.to_datetime(upload_df["timestamp"])
                
            # Preview data
            st.dataframe(upload_df.head())
            
            if st.button("Confirm Upload to Database"):
                with st.spinner("Uploading data..."):
                    # Insert data into database
                    conn = sqlite3.connect(DB_PATH)
                    upload_df.to_sql('trades', conn, if_exists='append', index=False)
                    conn.commit()
                    conn.close()
                    
                    st.success(f"Successfully uploaded {len(upload_df)} records to the database!")
                    st.experimental_rerun()
    except Exception as e:
        st.error(f"Error processing file: {e}")