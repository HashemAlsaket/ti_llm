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

def init_db():
    """Initialize SQLite database and create tables if they don't exist"""
    
    # Let's try to use multiple possible locations
    possible_paths = [
        os.path.join(tempfile.gettempdir(), "finance_data.db"),  # Temp directory
        os.path.join(os.path.expanduser("~"), "finance_data.db"),  # Home directory
        "finance_data.db"  # Current directory (original)
    ]
    
    # Try each location until one works
    for path in possible_paths:
        try:
            # Print the path we're trying
            print(f"Attempting to create database at: {path}")
            
            # Remove if exists
            if os.path.exists(path):
                print(f"Removing existing database at {path}")
                os.remove(path)
                
            # Try to create and open the database
            conn = sqlite3.connect(path)
            cursor = conn.cursor()
            
            # Test write access with a simple query
            cursor.execute("CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY)")
            conn.commit()
            
            print(f"Success! Using database at: {path}")
            
            # If we get here, we've found a working path
            global DB_PATH
            DB_PATH = path
            
            # Continue with your original table creation
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
            
            # Create commodities table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS commodities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                commodity_name TEXT NOT NULL,
                ticker TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                price REAL NOT NULL,
                volume INTEGER,
                change_pct REAL,
                inventory_level REAL,
                category TEXT
            )
            ''')
            
            # Create interest_rates table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS interest_rates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rate_name TEXT NOT NULL,
                country TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                rate_value REAL NOT NULL,
                previous_value REAL,
                term TEXT,
                is_central_bank BOOLEAN
            )
            ''')
            
            # Create real_estate table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS real_estate (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                property_type TEXT NOT NULL,
                region TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                price_index REAL NOT NULL,
                yoy_change_pct REAL,
                inventory_level INTEGER,
                avg_days_on_market INTEGER
            )
            ''')
            
            # Create economic_indicators table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS economic_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                indicator_name TEXT NOT NULL,
                country TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                value REAL NOT NULL,
                previous_value REAL,
                unit TEXT,
                frequency TEXT
            )
            ''')
            
            conn.commit()
            conn.close()
            
            # Success - break out of the loop
            break
            
        except Exception as e:
            print(f"Error with path {path}: {str(e)}")
            continue
    else:
        # If we get here, none of the paths worked
        st.error("Could not initialize database. Please check permissions and disk space.")
        raise Exception("Failed to initialize database after trying multiple locations")

def db_is_empty():
    """Check if database is empty"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check main tables
    cursor.execute("SELECT COUNT(*) FROM trades")
    trades_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='commodities'")
    commodities_table_exists = cursor.fetchone()[0] > 0
    
    if commodities_table_exists:
        cursor.execute("SELECT COUNT(*) FROM commodities")
        commodities_count = cursor.fetchone()[0]
    else:
        commodities_count = 0
        
    conn.close()
    return trades_count == 0 or commodities_count == 0

def load_data_to_db():
    """Generate mock data and load into SQLite database"""
    np.random.seed(42)
    
    # --- TRADES DATA ---
    # Enhanced tickers list including various asset classes - REDUCED
    tickers = {
        'Equities': ['AAPL', 'MSFT', 'GOOG'],
        'ETFs': ['SPY', 'QQQ'],
        'Bonds': ['TLT', 'IEF'],
        'Forex': ['EUR/USD', 'GBP/USD'],
        'Crypto': ['BTC/USD', 'ETH/USD']
    }
    
    # Enhanced model groups - REDUCED
    model_groups = ['Macro Alpha', 'Q1 Equity', 'Commodities Signal', 'Rates Momentum']
    
    # Enhanced sectors - REDUCED
    sectors = ['Technology', 'Healthcare', 'Financials', 'Energy', 'Materials']
    
    # Flatten tickers for random selection
    all_tickers = []
    for asset_class, ticker_list in tickers.items():
        for ticker in ticker_list:
            all_tickers.append((ticker, asset_class))
    
    trades_data = []
    # REDUCED number of trades from 1000 to 50
    for _ in range(50):
        chosen_index = np.random.randint(0, len(all_tickers))
        ticker, asset_class = all_tickers[chosen_index]
        
        # Assign sectors based on ticker (simplified logic)
        if ticker in ['AAPL', 'MSFT', 'GOOG']:
            sector = 'Technology'
        elif ticker in ['JPM', 'BAC'] or asset_class == 'Bonds':
            sector = 'Financials'
        else:
            # Fix for numpy.random.choice
            sector = sectors[np.random.randint(0, len(sectors))]
        
        # Generate the trade data
        trades_data.append({
            "ticker": ticker,
            "model_group": model_groups[np.random.randint(0, len(model_groups))],
            "timestamp": datetime(2024, np.random.randint(1, 3), np.random.randint(1, 15)),
            "position": np.random.uniform(-2000000, 2000000),
            "pnl": np.random.uniform(-200000, 300000),
            "alpha_score": np.random.normal(0, 1),
            "volatility": np.random.uniform(0.05, 0.5),
            "sector": sector,
            "asset_class": asset_class
        })
    
    # --- COMMODITIES DATA --- REDUCED
    commodities = [
        # Reduced to just a few commodities
        ('Crude Oil WTI', 'CL=F', 'Energy'),
        ('Gold', 'GC=F', 'Precious Metals'),
        ('Corn', 'ZC=F', 'Agriculture')
    ]
    
    commodities_data = []
    
    # Generate 1 month of data (instead of 3 months)
    for commodity_name, ticker, category in commodities:
        # Set base price based on commodity
        if category == 'Energy':
            base_price = 70.0
        elif category == 'Precious Metals':
            base_price = 1900.0
        else:  # Agriculture
            base_price = 10.0
        
        # Reduced from 90 days to 15 days
        for days_ago in range(15, 0, -1):
            date = datetime.now() - timedelta(days=days_ago)
            daily_change = np.random.normal(0, 0.02)
            price = base_price * (1 + daily_change)
            
            commodities_data.append({
                "commodity_name": commodity_name,
                "ticker": ticker,
                "timestamp": date,
                "price": price,
                "volume": int(np.random.uniform(50000, 200000)),
                "change_pct": daily_change * 100,
                "inventory_level": np.random.uniform(100000, 400000),
                "category": category
            })
    
    # --- INTEREST RATES DATA --- REDUCED
    interest_rates_info = [
        # Just a few key rates
        ('Federal Funds Rate', 'US', 'Overnight', True),
        ('ECB Main Refinancing Rate', 'EU', 'Overnight', True),
        ('US Treasury Yield', 'US', '10-Year', False)
    ]
    
    interest_rates_data = []
    
    for rate_name, country, term, is_central_bank in interest_rates_info:
        # Simplified rate assignment
        if is_central_bank:
            base_rate = 4.5 if country == 'US' else 3.75
        else:
            base_rate = 4.0
        
        # Reduced from 12 months to 4 months
        for month in range(4, 0, -1):
            date = datetime.now().replace(day=15) - timedelta(days=month*30)
            
            # Previous month's value
            if month == 4:
                previous_value = base_rate - 0.1
            else:
                previous_value = current_value
            
            # Current month's value - simplified
            rate_change = 0.05 if np.random.random() > 0.7 else 0
            current_value = previous_value + rate_change
            
            interest_rates_data.append({
                "rate_name": rate_name,
                "country": country,
                "timestamp": date,
                "rate_value": current_value,
                "previous_value": previous_value,
                "term": term,
                "is_central_bank": 1 if is_central_bank else 0
            })
    
    # --- REAL ESTATE DATA --- REDUCED
    real_estate_info = [
        # Just a few key property types
        ('Single Family', 'US-National', 'Residential'),
        ('Office', 'US-National', 'Commercial')
    ]
    
    real_estate_data = []
    
    for property_type, region, category in real_estate_info:
        base_index = 200.0 if category == 'Residential' else 180.0
        
        # Just 2 quarters instead of 8
        for quarter in range(2, 0, -1):
            date = datetime(2024, quarter * 3, 15)
            price_index = base_index * (1 + (quarter * 0.01))
            yoy_change = 2.5  # Simplified
            avg_days = 45 if category == 'Residential' else 90
            
            real_estate_data.append({
                "property_type": property_type,
                "region": region,
                "timestamp": date,
                "price_index": price_index,
                "yoy_change_pct": yoy_change,
                "inventory_level": 10000,
                "avg_days_on_market": avg_days
            })
    
    # --- ECONOMIC INDICATORS DATA --- REDUCED
    economic_indicators_info = [
        # Just a few key indicators
        ('GDP Growth Rate', 'US', '%', 'Quarterly'),
        ('CPI', 'US', '%', 'Monthly'),
        ('Unemployment Rate', 'US', '%', 'Monthly')
    ]
    
    economic_data = []
    
    for indicator_name, country, unit, frequency in economic_indicators_info:
        # Simplified values
        if indicator_name == 'GDP Growth Rate':
            base_value = 2.5
        elif indicator_name == 'CPI':
            base_value = 3.0
        else:  # Unemployment
            base_value = 4.0
        
        # Reduced from 24/8 periods to just 3
        periods = 3
        days_per_period = 30 if frequency == 'Monthly' else 90
        
        for period in range(periods, 0, -1):
            date = datetime.now() - timedelta(days=period*days_per_period)
            
            if period == periods:
                previous_value = base_value - 0.1
            else:
                previous_value = current_value
            
            current_value = previous_value + np.random.normal(0, 0.1)
            
            economic_data.append({
                "indicator_name": indicator_name,
                "country": country,
                "timestamp": date,
                "value": current_value,
                "previous_value": previous_value,
                "unit": unit,
                "frequency": frequency
            })
    
    # Insert data into database
    conn = sqlite3.connect(DB_PATH)
    
    # Insert trades data
    trades_df = pd.DataFrame(trades_data)
    trades_df.to_sql('trades', conn, if_exists='append', index=False)
    
    # Insert commodities data
    commodities_df = pd.DataFrame(commodities_data)
    commodities_df.to_sql('commodities', conn, if_exists='append', index=False)
    
    # Insert interest rates data
    interest_rates_df = pd.DataFrame(interest_rates_data)
    interest_rates_df.to_sql('interest_rates', conn, if_exists='append', index=False)
    
    # Insert real estate data
    real_estate_df = pd.DataFrame(real_estate_data)
    real_estate_df.to_sql('real_estate', conn, if_exists='append', index=False)
    
    # Insert economic indicators data
    economic_df = pd.DataFrame(economic_data)
    economic_df.to_sql('economic_indicators', conn, if_exists='append', index=False)
    
    conn.commit()
    conn.close()
    
    return trades_df

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