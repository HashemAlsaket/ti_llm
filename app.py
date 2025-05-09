import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
import tempfile
DB_PATH = os.path.join(tempfile.gettempdir(), "finance_data.db")
# --- DATABASE SETUP ---
# Use an in-memory database instead of a file
DB_PATH = ":memory:"

def init_db():
    """Initialize in-memory SQLite database and create tables"""
    try:
        # Connect to in-memory database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Print for debugging
        print("Connected to in-memory database")
        
        # Create trades table (minimal)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            model_group TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            position REAL NOT NULL,
            pnl REAL NOT NULL,
            alpha_score REAL NOT NULL,
            volatility REAL NOT NULL,
            sector TEXT,
            asset_class TEXT
        )
        ''')
        
        # Only create the essential tables (removing others to minimize complexity)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS commodities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            commodity_name TEXT NOT NULL,
            ticker TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            price REAL NOT NULL,
            category TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
        print("Database initialization successful!")
        
        # Store a flag in session state to indicate the database is initialized
        st.session_state["db_initialized"] = True
        
    except Exception as e:
        print(f"Database initialization error: {str(e)}")
        st.error(f"Could not initialize database: {str(e)}")
        raise e

def get_connection():
    """Get a connection to the in-memory database"""
    if "db_conn" not in st.session_state:
        st.session_state.db_conn = sqlite3.connect(DB_PATH)
    return st.session_state.db_conn

def db_is_empty():
    """Check if database is empty"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check trades table
        cursor.execute("SELECT COUNT(*) FROM trades")
        trades_count = cursor.fetchone()[0]
        
        return trades_count == 0
    except Exception as e:
        print(f"Error checking if DB is empty: {str(e)}")
        return True

def load_data_to_db():
    """Generate minimal mock data and load into SQLite database"""
    try:
        conn = get_connection()
        
        # Just 5 trades
        trades_data = [
            {"ticker": "AAPL", "model_group": "Macro Alpha", "timestamp": datetime.now(), "position": 1000000, "pnl": 50000, "alpha_score": 0.5, "volatility": 0.2, "sector": "Technology", "asset_class": "Equities"},
            {"ticker": "MSFT", "model_group": "Q1 Equity", "timestamp": datetime.now(), "position": 1500000, "pnl": 75000, "alpha_score": 0.6, "volatility": 0.25, "sector": "Technology", "asset_class": "Equities"},
            {"ticker": "SPY", "model_group": "Rates Momentum", "timestamp": datetime.now(), "position": -500000, "pnl": -25000, "alpha_score": -0.3, "volatility": 0.15, "sector": "Financials", "asset_class": "ETFs"},
            {"ticker": "TLT", "model_group": "Rates Momentum", "timestamp": datetime.now(), "position": 800000, "pnl": 40000, "alpha_score": 0.4, "volatility": 0.1, "sector": "Financials", "asset_class": "Bonds"},
            {"ticker": "BTC/USD", "model_group": "Technical Breakout", "timestamp": datetime.now(), "position": 300000, "pnl": 100000, "alpha_score": 0.8, "volatility": 0.4, "sector": "Technology", "asset_class": "Crypto"}
        ]
        
        # Just 3 commodities
        commodities_data = [
            {"commodity_name": "Crude Oil WTI", "ticker": "CL=F", "timestamp": datetime.now(), "price": 75.0, "category": "Energy"},
            {"commodity_name": "Gold", "ticker": "GC=F", "timestamp": datetime.now(), "price": 1900.0, "category": "Precious Metals"},
            {"commodity_name": "Corn", "ticker": "ZC=F", "timestamp": datetime.now(), "price": 12.0, "category": "Agriculture"}
        ]
        
        # Insert data directly, simplifying the process
        trades_df = pd.DataFrame(trades_data)
        trades_df.to_sql('trades', conn, if_exists='append', index=False)
        
        commodities_df = pd.DataFrame(commodities_data)
        commodities_df.to_sql('commodities', conn, if_exists='append', index=False)
        
        conn.commit()
        print("Data loaded successfully!")
        
        return trades_df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        st.error(f"Failed to load sample data: {str(e)}")
        return pd.DataFrame()

def get_data_from_db(ticker="All", model_group="All"):
    """Fetch data from SQLite database with optional filters"""
    try:
        conn = get_connection()
        
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
        return df
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def get_detailed_db_schema():
    """Get a detailed database schema description for better LLM context"""
    schema = """
Table: trades
Description: This table contains financial trading data for various financial instruments across different trading models and asset classes.

Columns:
* id (INTEGER) # Unique identifier for each trade record
* ticker (TEXT) # Symbol representing the instrument being traded (stocks, ETFs, forex, crypto, etc.)
* model_group (TEXT) # The trading strategy or model used for the trade (e.g., Macro Alpha, Q1 Equity)
* timestamp (TIMESTAMP) # Date and time when the trade was executed or recorded
* position (REAL) # Current position size in dollars; positive values indicate long positions, negative values indicate short positions
* pnl (REAL) # Profit and Loss in dollars; indicates how much profit (positive) or loss (negative) the position has generated
* alpha_score (REAL) # A score measuring the excess return of the investment relative to a benchmark; higher values indicate better performance
* volatility (REAL) # A measure of the price variation/risk of the position; higher values indicate more volatile/risky positions
* sector (TEXT) # The market sector of the instrument (e.g., Technology, Energy, Financials)
* asset_class (TEXT) # The class of financial asset (Equities, ETFs, Bonds, Forex, Crypto)

Table: commodities
Description: This table contains price and market data for various commodities including energy products, metals, and agricultural goods.

Columns:
* id (INTEGER) # Unique identifier for each commodity data point
* commodity_name (TEXT) # Name of the commodity (e.g., Crude Oil WTI, Gold, Corn)
* ticker (TEXT) # Trading symbol for the commodity (e.g., CL=F, GC=F)
* timestamp (TIMESTAMP) # Date and time of the commodity data point
* price (REAL) # Price of the commodity in USD
* volume (INTEGER) # Trading volume of the commodity
* change_pct (REAL) # Percentage change in price from the previous period
* inventory_level (REAL) # Available inventory/stockpile of the commodity
* category (TEXT) # Category of the commodity (Energy, Precious Metals, Base Metals, Agriculture)

Table: interest_rates
Description: This table contains interest rate data for various countries, terms, and rate types including central bank rates and market yields.

Columns:
* id (INTEGER) # Unique identifier for each interest rate data point
* rate_name (TEXT) # Name of the interest rate (e.g., Federal Funds Rate, US Treasury Yield)
* country (TEXT) # Country or region associated with the rate (e.g., US, EU, UK)
* timestamp (TIMESTAMP) # Date and time of the interest rate data point
* rate_value (REAL) # The interest rate value in percentage
* previous_value (REAL) # Previous period's interest rate value
* term (TEXT) # Duration/term of the rate (e.g., Overnight, 2-Year, 10-Year)
* is_central_bank (BOOLEAN) # Flag indicating if the rate is set by a central bank (1) or market-determined (0)

Table: real_estate
Description: This table contains real estate market data across different property types and regions.

Columns:
* id (INTEGER) # Unique identifier for each real estate data point
* property_type (TEXT) # Type of property (e.g., Single Family, Office, Retail)
* region (TEXT) # Geographic region (e.g., US-National, US-Northeast, UK)
* timestamp (TIMESTAMP) # Date and time of the real estate data point
* price_index (REAL) # Price index value representing property values
* yoy_change_pct (REAL) # Year-over-year percentage change in price
* inventory_level (INTEGER) # Available inventory of properties
* avg_days_on_market (INTEGER) # Average number of days properties remain on market before sale

Table: economic_indicators
Description: This table contains economic data for various countries including growth metrics, inflation figures, and employment statistics.

Columns:
* id (INTEGER) # Unique identifier for each economic indicator data point
* indicator_name (TEXT) # Name of the economic indicator (e.g., GDP Growth Rate, CPI, Unemployment Rate)
* country (TEXT) # Country or region associated with the indicator (e.g., US, EU, UK)
* timestamp (TIMESTAMP) # Date and time of the economic indicator data point
* value (REAL) # Value of the economic indicator
* previous_value (REAL) # Previous period's value for the economic indicator
* unit (TEXT) # Unit of measurement (e.g., %, Index, Thousands)
* frequency (TEXT) # Frequency of data collection (e.g., Monthly, Quarterly)
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
    with st.spinner("Loading financial market data..."):
        load_data_to_db()
        st.success("Financial data loaded successfully!")

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
            "Consider which tables might contain the data needed to answer the question."
            "Use appropriate aggregation functions like AVG, SUM, COUNT as needed."
            "For date comparisons, remember SQLite timestamp format."
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
        """
        )
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

# --- DATA SELECTION ---
table_options = ["trades", "commodities", "interest_rates", "real_estate", "economic_indicators"]
selected_table = st.sidebar.selectbox("Select Data Table", table_options)

# --- SIDEBAR FILTERS ---
st.sidebar.markdown("<p class='section-header'>🔍 Filter Data</p>", unsafe_allow_html=True)

# Dynamic filters based on selected table
if selected_table == "trades":
    ticker_options = ["All"]
    model_options = ["All"]
    
    # Get unique values from the database
    conn = sqlite3.connect(DB_PATH)
    ticker_options += [row[0] for row in conn.execute(f"SELECT DISTINCT ticker FROM {selected_table} ORDER BY ticker").fetchall()]
    model_options += [row[0] for row in conn.execute(f"SELECT DISTINCT model_group FROM {selected_table} ORDER BY model_group").fetchall()]
    conn.close()
    
    ticker = st.sidebar.selectbox("Select Ticker", ticker_options)
    model = st.sidebar.selectbox("Select Model Group", model_options)
    
    # Get filtered data
    filtered_df = get_data_from_db(ticker, model)
    
elif selected_table == "commodities":
    category_options = ["All"]
    commodity_options = ["All"]
    
    # Get unique values
    conn = sqlite3.connect(DB_PATH)
    category_options += [row[0] for row in conn.execute(f"SELECT DISTINCT category FROM {selected_table} ORDER BY category").fetchall()]
    commodity_options += [row[0] for row in conn.execute(f"SELECT DISTINCT commodity_name FROM {selected_table} ORDER BY commodity_name").fetchall()]
    conn.close()
    
    category = st.sidebar.selectbox("Select Category", category_options)
    commodity = st.sidebar.selectbox("Select Commodity", commodity_options)
    
    # Build query
    query = f"SELECT * FROM {selected_table}"
    params = []
    where_clauses = []
    
    if category != "All":
        where_clauses.append("category = ?")
        params.append(category)
    if commodity != "All":
        where_clauses.append("commodity_name = ?")
        params.append(commodity)
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    # Get data
    conn = sqlite3.connect(DB_PATH)
    filtered_df = pd.read_sql(query, conn, params=params)
    conn.close()
    
    # Set defaults for trades table filters
    ticker = "All"
    model = "All"

elif selected_table == "interest_rates":
    country_options = ["All"]
    rate_type_options = ["All"]
    
    # Get unique values
    conn = sqlite3.connect(DB_PATH)
    country_options += [row[0] for row in conn.execute(f"SELECT DISTINCT country FROM {selected_table} ORDER BY country").fetchall()]
    rate_type_options += [row[0] for row in conn.execute(f"SELECT DISTINCT rate_name FROM {selected_table} ORDER BY rate_name").fetchall()]
    conn.close()
    
    country = st.sidebar.selectbox("Select Country", country_options)
    rate_type = st.sidebar.selectbox("Select Rate Type", rate_type_options)
    
    # Build query
    query = f"SELECT * FROM {selected_table}"
    params = []
    where_clauses = []
    
    if country != "All":
        where_clauses.append("country = ?")
        params.append(country)
    if rate_type != "All":
        where_clauses.append("rate_name = ?")
        params.append(rate_type)
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    # Get data
    conn = sqlite3.connect(DB_PATH)
    filtered_df = pd.read_sql(query, conn, params=params)
    conn.close()
    
    # Set defaults for trades table filters
    ticker = "All"
    model = "All"

elif selected_table == "real_estate":
    region_options = ["All"]
    property_type_options = ["All"]
    
    # Get unique values
    conn = sqlite3.connect(DB_PATH)
    region_options += [row[0] for row in conn.execute(f"SELECT DISTINCT region FROM {selected_table} ORDER BY region").fetchall()]
    property_type_options += [row[0] for row in conn.execute(f"SELECT DISTINCT property_type FROM {selected_table} ORDER BY property_type").fetchall()]
    conn.close()
    
    region = st.sidebar.selectbox("Select Region", region_options)
    property_type = st.sidebar.selectbox("Select Property Type", property_type_options)
    
    # Build query
    query = f"SELECT * FROM {selected_table}"
    params = []
    where_clauses = []
    
    if region != "All":
        where_clauses.append("region = ?")
        params.append(region)
    if property_type != "All":
        where_clauses.append("property_type = ?")
        params.append(property_type)
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    # Get data
    conn = sqlite3.connect(DB_PATH)
    filtered_df = pd.read_sql(query, conn, params=params)
    conn.close()
    
    # Set defaults for trades table filters
    ticker = "All"
    model = "All"

elif selected_table == "economic_indicators":
    country_options = ["All"]
    indicator_options = ["All"]
    
    # Get unique values
    conn = sqlite3.connect(DB_PATH)
    country_options += [row[0] for row in conn.execute(f"SELECT DISTINCT country FROM {selected_table} ORDER BY country").fetchall()]
    indicator_options += [row[0] for row in conn.execute(f"SELECT DISTINCT indicator_name FROM {selected_table} ORDER BY indicator_name").fetchall()]
    conn.close()
    
    country = st.sidebar.selectbox("Select Country", country_options)
    indicator = st.sidebar.selectbox("Select Indicator", indicator_options)
    
    # Build query
    query = f"SELECT * FROM {selected_table}"
    params = []
    where_clauses = []
    
    if country != "All":
        where_clauses.append("country = ?")
        params.append(country)
    if indicator != "All":
        where_clauses.append("indicator_name = ?")
        params.append(indicator)
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    # Get data
    conn = sqlite3.connect(DB_PATH)
    filtered_df = pd.read_sql(query, conn, params=params)
    conn.close()
    
    # Set defaults for trades table filters
    ticker = "All"
    model = "All"

# --- QUERY APPROACH SELECTION ---
st.sidebar.markdown("<p class='section-header'>💡 Query Method</p>", unsafe_allow_html=True)
query_method = st.sidebar.radio(
    "LLM Query Method",
    ["Simple (Direct)", "SQL-Based", "Enhanced SQL"],
    help="Choose how the LLM will process your query"
)

# --- SAMPLE QUESTIONS ---
st.sidebar.markdown("<p class='section-header'>❓ Sample Questions</p>", unsafe_allow_html=True)

# Dynamic sample questions based on selected table
if selected_table == "trades":
    sample_questions = [
        "Which model group has the highest total PnL?",
        "What is the relationship between alpha score and volatility?",
        "Compare the average position size across different tickers",
        "Which ticker has the most negative position?",
        "Show me the trading performance by model group",
        "Which asset class has performed the best based on PnL?",
        "How does sector exposure correlate with volatility?",
        "What is our exposure to the energy sector vs technology sector?"
    ]
elif selected_table == "commodities":
    sample_questions = [
        "What is the trend in oil prices over the last 3 months?",
        "Compare the price volatility of gold versus silver",
        "Which commodity category has the highest average inventory levels?",
        "How have agricultural commodities performed compared to metals?",
        "What is the correlation between oil prices and inventory levels?",
        "Show me the price trends of energy commodities",
        "Which commodity had the largest price increase in the past month?"
    ]
elif selected_table == "interest_rates":
    sample_questions = [
        "How have central bank rates changed over the past year?",
        "Compare the 10-year yields across different countries",
        "What is the spread between 2-year and 10-year US Treasury yields?",
        "Which country has the highest current interest rates?",
        "Show the trend of Federal Funds Rate over time",
        "Is there evidence of yield curve inversion in US rates?",
        "Compare LIBOR rates to central bank rates"
    ]
elif selected_table == "real_estate":
    sample_questions = [
        "Which region has seen the highest price appreciation in residential real estate?",
        "Compare commercial vs residential property performance",
        "What is the trend in US national home prices?",
        "Which property type has the shortest average days on market?",
        "How do US and UK residential markets compare?",
        "Show the inventory levels across different regions",
        "Which property type has been most volatile in price?"
    ]
elif selected_table == "economic_indicators":
    sample_questions = [
        "Compare GDP growth rates across major economies",
        "What is the relationship between inflation and unemployment in the US?",
        "How has US consumer sentiment changed over time?",
        "Which country has the highest inflation rate?",
        "Show the trend in US non-farm payrolls",
        "Compare manufacturing PMI across countries",
        "Is there evidence of stagflation in any economies?"
    ]
else:
    sample_questions = [
        "Which model group has the highest total PnL?",
        "What is the trend in oil prices over the last 3 months?",
        "How have central bank rates changed over the past year?",
        "Which region has seen the highest price appreciation in residential real estate?",
        "Compare GDP growth rates across major economies"
    ]

# Cross-table complex questions
complex_questions = [
    "How do changes in interest rates correlate with equity performance?",
    "What is the relationship between oil prices and energy sector stocks?",
    "How does GDP growth relate to real estate price trends?",
    "Compare inflation rates with gold price movements",
    "Analyze the impact of unemployment rate changes on financial sector performance",
    "How do real estate trends correlate with interest rate movements?",
    "What is the relationship between consumer sentiment and retail sector performance?"
]

# Add complex questions that cross multiple tables
sample_questions.extend(complex_questions)

selected_question = st.sidebar.selectbox("Try a sample question:", [""] + sample_questions)

# --- DISPLAY DATA ---
st.markdown(f"<p class='section-header'>📊 {selected_table.capitalize()} Data</p>", unsafe_allow_html=True)
st.dataframe(filtered_df, use_container_width=True)

# --- SUMMARY METRICS (for selected table) ---
col1, col2, col3 = st.columns(3)

if selected_table == "trades":
    with col1:
        st.metric("Total PnL", f"${filtered_df['pnl'].sum():,.0f}")
    with col2:
        st.metric("Net Position", f"${filtered_df['position'].sum():,.0f}")
    with col3:
        st.metric("Avg Alpha Score", f"{filtered_df['alpha_score'].mean():.2f}")

elif selected_table == "commodities":
    with col1:
        st.metric("Avg Price", f"${filtered_df['price'].mean():.2f}")
    with col2:
        st.metric("Avg Change", f"{filtered_df['change_pct'].mean():.2f}%")
    with col3:
        st.metric("Total Volume", f"{filtered_df['volume'].sum():,.0f}")

elif selected_table == "interest_rates":
    with col1:
        st.metric("Avg Rate", f"{filtered_df['rate_value'].mean():.2f}%")
    with col2:
        central_bank_rates = filtered_df[filtered_df['is_central_bank'] == 1]['rate_value']
        if not central_bank_rates.empty:
            st.metric("Avg Central Bank Rate", f"{central_bank_rates.mean():.2f}%")
        else:
            st.metric("Avg Central Bank Rate", "N/A")
    with col3:
        market_rates = filtered_df[filtered_df['is_central_bank'] == 0]['rate_value']
        if not market_rates.empty:
            st.metric("Avg Market Rate", f"{market_rates.mean():.2f}%")
        else:
            st.metric("Avg Market Rate", "N/A")

elif selected_table == "real_estate":
    with col1:
        st.metric("Avg Price Index", f"{filtered_df['price_index'].mean():.1f}")
    with col2:
        st.metric("Avg YoY Change", f"{filtered_df['yoy_change_pct'].mean():.2f}%")
    with col3:
        st.metric("Avg Days on Market", f"{filtered_df['avg_days_on_market'].mean():.0f}")

elif selected_table == "economic_indicators":
    # Different metrics based on unit
    percent_indicators = filtered_df[filtered_df['unit'] == '%']
    index_indicators = filtered_df[filtered_df['unit'] == 'Index']
    other_indicators = filtered_df[~filtered_df['unit'].isin(['%', 'Index'])]
    
    with col1:
        if not percent_indicators.empty:
            st.metric("Avg % Indicators", f"{percent_indicators['value'].mean():.2f}%")
        else:
            st.metric("Avg % Indicators", "N/A")
    with col2:
        if not index_indicators.empty:
            st.metric("Avg Index Value", f"{index_indicators['value'].mean():.1f}")
        else:
            st.metric("Avg Index Value", "N/A")
    with col3:
        if not other_indicators.empty:
            st.metric("Data Points", f"{len(filtered_df)}")
        else:
            st.metric("Data Points", f"{len(filtered_df)}")

# --- SCHEMA DISPLAY ---
with st.expander("View Database Schema"):
    st.code(get_detailed_db_schema(), language="markdown")

# --- LLM QUERY INTERFACE ---
st.markdown("<p class='section-header'>🤖 Ask a Question</p>", unsafe_allow_html=True)

# Use the selected sample question if one is chosen
if selected_question:
    user_prompt = st.text_area("What would you like to know about this financial data?", 
                             value=selected_question,
                             height=100,
                             placeholder="Example: Which model group has the highest PnL?")
else:
    user_prompt = st.text_area("What would you like to know about this financial data?", 
                             height=100,
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

# Select table for upload
upload_table = st.selectbox("Select table to upload data to:", table_options)

uploaded_file = st.file_uploader(f"Upload CSV file with {upload_table} data", type=["csv"])
if uploaded_file is not None:
    # Read uploaded CSV
    try:
        upload_df = pd.read_csv(uploaded_file)
        
        # Get column names for the selected table
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({upload_table});")
        table_columns = [col[1] for col in cursor.fetchall() if col[1] != 'id']
        conn.close()
        
        # Check if required columns exist
        missing_cols = [col for col in table_columns if col not in upload_df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
        else:
            # Convert timestamp if needed
            if 'timestamp' in upload_df.columns and not pd.api.types.is_datetime64_any_dtype(upload_df["timestamp"]):
                upload_df["timestamp"] = pd.to_datetime(upload_df["timestamp"])
                
            # Preview data
            st.dataframe(upload_df.head())
            
            if st.button(f"Confirm Upload to {upload_table} Table"):
                with st.spinner("Uploading data..."):
                    # Insert data into database
                    conn = sqlite3.connect(DB_PATH)
                    upload_df.to_sql(upload_table, conn, if_exists='append', index=False)
                    conn.commit()
                    conn.close()
                    
                    st.success(f"Successfully uploaded {len(upload_df)} records to {upload_table}!")
                    st.experimental_rerun()
    except Exception as e:
        st.error(f"Error processing file: {e}")