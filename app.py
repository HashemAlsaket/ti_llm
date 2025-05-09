import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os
import tempfile
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
# Use a temporary file location that should have write access
temp_dir = tempfile.gettempdir()
DB_PATH = os.path.join(temp_dir, "finance_data.db")
print(f"Using database at: {DB_PATH}")

def init_db():
    """Initialize SQLite database with minimal tables"""
    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
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
        
        # Only create one additional table to keep things simple
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
        
        # Insert minimal sample data immediately
        sample_trades = [
            ("AAPL", "Macro Alpha", "2024-01-15", 1000000, 50000, 0.5, 0.2, "Technology", "Equities"),
            ("MSFT", "Q1 Equity", "2024-01-15", 1500000, 75000, 0.6, 0.25, "Technology", "Equities"),
            ("SPY", "Rates Momentum", "2024-01-15", -500000, -25000, -0.3, 0.15, "Financials", "ETFs"),
            ("BTC/USD", "Technical Breakout", "2024-01-15", 300000, 100000, 0.8, 0.4, "Technology", "Crypto")
        ]
        
        # Check if data already exists
        cursor.execute("SELECT COUNT(*) FROM trades")
        count = cursor.fetchone()[0]
        
        if count == 0:
            cursor.executemany('''
                INSERT INTO trades (ticker, model_group, timestamp, position, pnl, alpha_score, volatility, sector, asset_class)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', sample_trades)
            
            # Add sample commodities data
            sample_commodities = [
                ("Gold", "GC=F", "2024-01-15", 1900.0, "Precious Metals"),
                ("Crude Oil WTI", "CL=F", "2024-01-15", 75.0, "Energy")
            ]
            
            cursor.executemany('''
                INSERT INTO commodities (commodity_name, ticker, timestamp, price, category)
                VALUES (?, ?, ?, ?, ?)
            ''', sample_commodities)
        
        conn.commit()
        conn.close()
        print(f"Database initialized at {DB_PATH} with {len(sample_trades)} sample trades")
        
    except Exception as e:
        print(f"Database initialization error: {str(e)}")
        st.error(f"Could not initialize database: {str(e)}")

def db_is_empty():
    """Check if database is empty"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if trades table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
        if not cursor.fetchone():
            conn.close()
            return True
        
        # Check if trades table has data
        cursor.execute("SELECT COUNT(*) FROM trades")
        count = cursor.fetchone()[0]
        conn.close()
        
        return count == 0
    except Exception as e:
        print(f"Error checking if database is empty: {str(e)}")
        return True

def load_data_to_db():
    """Generate minimal mock data and load into SQLite database"""
    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        
        # Minimal trades data
        trades_data = [
            {"ticker": "AAPL", "model_group": "Macro Alpha", "timestamp": "2024-01-15", 
             "position": 1000000, "pnl": 50000, "alpha_score": 0.5, "volatility": 0.2, 
             "sector": "Technology", "asset_class": "Equities"},
            {"ticker": "MSFT", "model_group": "Q1 Equity", "timestamp": "2024-01-15", 
             "position": 1500000, "pnl": 75000, "alpha_score": 0.6, "volatility": 0.25, 
             "sector": "Technology", "asset_class": "Equities"},
            {"ticker": "SPY", "model_group": "Rates Momentum", "timestamp": "2024-01-15", 
             "position": -500000, "pnl": -25000, "alpha_score": -0.3, "volatility": 0.15, 
             "sector": "Financials", "asset_class": "ETFs"},
            {"ticker": "TLT", "model_group": "Rates Momentum", "timestamp": "2024-01-15", 
             "position": 800000, "pnl": 40000, "alpha_score": 0.4, "volatility": 0.1, 
             "sector": "Financials", "asset_class": "Bonds"},
            {"ticker": "BTC/USD", "model_group": "Technical Breakout", "timestamp": "2024-01-15", 
             "position": 300000, "pnl": 100000, "alpha_score": 0.8, "volatility": 0.4, 
             "sector": "Technology", "asset_class": "Crypto"}
        ]
        
        # Minimal commodities data
        commodities_data = [
            {"commodity_name": "Crude Oil WTI", "ticker": "CL=F", "timestamp": "2024-01-15", 
             "price": 75.0, "category": "Energy"},
            {"commodity_name": "Gold", "ticker": "GC=F", "timestamp": "2024-01-15", 
             "price": 1900.0, "category": "Precious Metals"}
        ]
        
        # Insert data
        trades_df = pd.DataFrame(trades_data)
        trades_df.to_sql('trades', conn, if_exists='replace', index=False)
        
        commodities_df = pd.DataFrame(commodities_data)
        commodities_df.to_sql('commodities', conn, if_exists='replace', index=False)
        
        conn.commit()
        conn.close()
        print("Sample data loaded successfully")
        
        return trades_df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def get_data_from_db(ticker="All", model_group="All"):
    """Fetch data from SQLite database with optional filters"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        query = "SELECT * FROM trades"
        params = []
        
        # Add filters if specified
        where_clauses = []
        if ticker != "All" and ticker:
            where_clauses.append("ticker = ?")
            params.append(ticker)
        if model_group != "All" and model_group:
            where_clauses.append("model_group = ?")
            params.append(model_group)
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        # Return empty DataFrame on error
        return pd.DataFrame(columns=["ticker", "model_group", "timestamp", "position", "pnl", "alpha_score", "volatility", "sector", "asset_class"])

def get_commodities_data(commodity="All", category="All"):
    """Fetch commodities data from SQLite database with optional filters"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        query = "SELECT * FROM commodities"
        params = []
        
        # Add filters if specified
        where_clauses = []
        if commodity != "All" and commodity:
            where_clauses.append("commodity_name = ?")
            params.append(commodity)
        if category != "All" and category:
            where_clauses.append("category = ?")
            params.append(category)
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df
    except Exception as e:
        print(f"Error fetching commodities data: {str(e)}")
        # Return empty DataFrame on error
        return pd.DataFrame(columns=["commodity_name", "ticker", "timestamp", "price", "category"])

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
* category (TEXT) # Category of the commodity (Energy, Precious Metals, Base Metals, Agriculture)
"""
    return schema

def get_db_schema():
    """Get the database schema as a string"""
    try:
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
    except Exception as e:
        print(f"Error fetching schema: {str(e)}")
        return "Could not retrieve database schema."

# --- LOGIN FLOW ---
def login():
    st.session_state["authenticated"] = False

    with st.form("Login"):
        st.write("🔐 Please log in to continue")
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            # For development purposes, accept any login
            # In production, replace with proper authentication
            if True: # Instead of checking credentials, always log in for simplicity
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
# Initialize the database at startup
try:
    init_db()
except Exception as e:
    st.error(f"Database initialization failed: {str(e)}")

# Load sample data if database is empty
if db_is_empty():
    with st.spinner("Loading minimal financial market data..."):
        try:
            load_data_to_db()
            st.success("Sample financial data loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load sample data: {str(e)}")

# --- INITIALIZE CLIENTS ---
try:
    # Skip API client initialization in development mode to avoid API key issues
    using_mock = True
    
    if using_mock:
        # Create mock clients that return placeholder responses
        class MockClient:
            def chat_completions_create(self, **kwargs):
                class Response:
                    class Message:
                        content = "This is a placeholder response. In production, this would come from the OpenAI API."
                    
                    class Choice:
                        def __init__(self):
                            self.message = Response.Message()
                    
                    def __init__(self):
                        self.choices = [Response.Choice()]
                
                return Response()
        
        openai_client = MockClient()
        langchain_llm = None
    else:
        # In production, use actual API clients
        openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        langchain_llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4")
        
except Exception as e:
    st.warning("Running in development mode with mock LLM responses. API clients not initialized.")
    
    class MockClient:
        def chat_completions_create(self, **kwargs):
            class Response:
                class Message:
                    content = "This is a placeholder response. In production, this would come from the OpenAI API."
                
                class Choice:
                    def __init__(self):
                        self.message = Response.Message()
                
                def __init__(self):
                    self.choices = [Response.Choice()]
            
            return Response()
    
    openai_client = MockClient()
    langchain_llm = None

# --- SETUP LANGCHAIN SQL DATABASE ---
try:
    db_url = f"sqlite:///{DB_PATH}"
    db = SQLDatabase.from_uri(db_url)
except Exception as e:
    st.warning(f"Could not initialize SQLDatabase: {str(e)}")
    db = None

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

    try:
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
    except Exception as e:
        print(f"Error querying LLM: {str(e)}")
        return "I'm sorry, I couldn't process that request due to an error. Please try again with a different question."

def ask_sql_llm(prompt):
    """Use LangChain to convert natural language to SQL and execute"""
    if not db or not langchain_llm:
        return {
            "explanation": "SQL-based queries are not available in development mode. Please switch to production mode to use this feature.",
            "sql": "-- SQL queries not available in development mode",
            "results": pd.DataFrame()
        }
    
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
    
    try:
        # In development mode, return mock response
        if using_mock:
            return {
                "explanation": "This is a mock response. In production, this would show analysis based on SQL results.",
                "sql": "SELECT ticker, SUM(pnl) AS total_pnl FROM trades GROUP BY ticker ORDER BY total_pnl DESC LIMIT 5",
                "results": pd.DataFrame({
                    "ticker": ["BTC/USD", "AAPL", "MSFT", "TLT", "SPY"],
                    "total_pnl": [100000, 50000, 75000, 40000, -25000]
                })
            }
        
        # Create a chain that generates SQL
        sql_chain = create_sql_query_chain(
            langchain_llm,
            db,
            prompt=sql_generation_prompt,
            k=3  # Number of examples used for few-shot prompting
        )
        
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
        print(f"Error in SQL LLM query: {str(e)}")
        return {
            "error": str(e),
            "sql": "Error generating or executing SQL",
            "explanation": "An error occurred while processing your question. Please try a different question or approach."
        }

def ask_enhanced_sql_llm(prompt, filtered_df):
    """Enhanced SQL-based approach with better context"""
    if not db or not langchain_llm:
        return {
            "response": "Enhanced SQL queries are not available in development mode. Please switch to production mode to use this feature.",
            "sql": "-- SQL queries not available in development mode",
            "sql_results": None
        }
    
    # In development mode, return mock response
    if using_mock:
        return {
            "response": "This is a mock response for the enhanced SQL approach. In production, this would show a detailed analysis combining SQL results with context.",
            "sql": "SELECT model_group, SUM(pnl) AS total_pnl FROM trades GROUP BY model_group ORDER BY total_pnl DESC",
            "sql_results": pd.DataFrame({
                "model_group": ["Technical Breakout", "Q1 Equity", "Macro Alpha", "Rates Momentum"],
                "total_pnl": [100000, 75000, 50000, 15000]
            })
        }
    
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
    
    try:
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
    except Exception as e:
        print(f"Error in enhanced SQL query: {str(e)}")
        return {
            "response": "An error occurred while processing your question with the enhanced approach. Please try a different question or approach.",
            "sql": sql_response.get("sql", "No SQL query generated"),
            "sql_results": None
        }

# --- DATA SELECTION ---
table_options = ["trades", "commodities"]  # Simplified to just two tables
selected_table = st.sidebar.selectbox("Select Data Table", table_options)

# --- SIDEBAR FILTERS ---
st.sidebar.markdown("<p class='section-header'>🔍 Filter Data</p>", unsafe_allow_html=True)

# Dynamic filters based on selected table
if selected_table == "trades":
    ticker_options = ["All"]
    model_options = ["All"]
    
    try:
        # Get unique values from the database
        conn = sqlite3.connect(DB_PATH)
        
        # Check if the table exists first
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
        if cursor.fetchone():
            ticker_options += [row[0] for row in conn.execute("SELECT DISTINCT ticker FROM trades ORDER BY ticker").fetchall()]
            model_options += [row[0] for row in conn.execute("SELECT DISTINCT model_group FROM trades ORDER BY model_group").fetchall()]
        else:
            st.warning("The trades table doesn't exist yet. Please initialize the database first.")
        
        conn.close()
    except Exception as e:
        st.error(f"Error loading filter options: {str(e)}")
        # Provide some default options if database query fails
        ticker_options = ["All", "AAPL", "MSFT", "SPY", "BTC/USD"]
        model_options = ["All", "Macro Alpha", "Q1 Equity", "Rates Momentum", "Technical Breakout"]
    
    ticker = st.sidebar.selectbox("Select Ticker", ticker_options)
    model = st.sidebar.selectbox("Select Model Group", model_options)
    
    # Get filtered data with error handling
    try:
        filtered_df = get_data_from_db(ticker, model)
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        # Provide empty dataframe if fetch fails
        filtered_df = pd.DataFrame(columns=["ticker", "model_group", "timestamp", "position", "pnl", "alpha_score", "volatility", "sector", "asset_class"])
    
elif selected_table == "commodities":
    category_options = ["All"]
    commodity_options = ["All"]
    
    try:
        # Get unique values
        conn = sqlite3.connect(DB_PATH)
        
        # Check if the table exists first
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='commodities'")
        if cursor.fetchone():
            category_options += [row[0] for row in conn.execute("SELECT DISTINCT category FROM commodities ORDER BY category").fetchall()]
            commodity_options += [row[0] for row in conn.execute("SELECT DISTINCT commodity_name FROM commodities ORDER BY commodity_name").fetchall()]
        else:
            st.warning("The commodities table doesn't exist yet. Please initialize the database first.")
        
        conn.close()
    except Exception as e:
        st.error(f"Error loading commodity filter options: {str(e)}")
        # Provide some default options if database query fails
        category_options = ["All", "Energy", "Precious Metals"]
        commodity_options = ["All", "Crude Oil WTI", "Gold"]
    
    category = st.sidebar.selectbox("Select Category", category_options)
    commodity = st.sidebar.selectbox("Select Commodity", commodity_options)
    
    # Get filtered data with error handling
    try:
        filtered_df = get_commodities_data(commodity, category)
    except Exception as e:
        st.error(f"Error fetching commodities data: {str(e)}")
        # Provide empty dataframe if fetch fails
        filtered_df = pd.DataFrame(columns=["commodity_name", "ticker", "timestamp", "price", "category"])
    
    # Set defaults for trades table filters for later use
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
        "Which ticker has the most negative position?"
    ]
elif selected_table == "commodities":
    sample_questions = [
        "What is the price of gold?",
        "Compare the prices of different commodities",
        "Which commodity category has the highest price?"
    ]
else:
    sample_questions = [
        "Which model group has the highest total PnL?",
        "What is the price of gold?"
    ]

selected_question = st.sidebar.selectbox("Try a sample question:", [""] + sample_questions)

# --- DISPLAY DATA ---
st.markdown(f"<p class='section-header'>📊 {selected_table.capitalize()} Data</p>", unsafe_allow_html=True)
st.dataframe(filtered_df, use_container_width=True)

# --- SUMMARY METRICS (for selected table) ---
col1, col2, col3 = st.columns(3)

if selected_table == "trades" and not filtered_df.empty:
    with col1:
        st.metric("Total PnL", f"${filtered_df['pnl'].sum():,.0f}")
    with col2:
        st.metric("Net Position", f"${filtered_df['position'].sum():,.0f}")
    with col3:
        st.metric("Avg Alpha Score", f"{filtered_df['alpha_score'].mean():.2f}")

elif selected_table == "commodities" and not filtered_df.empty:
    with col1:
        st.metric("Avg Price", f"${filtered_df['price'].mean():.2f}" if 'price' in filtered_df.columns else "N/A")
    with col2:
        st.metric("Categories", f"{filtered_df['category'].nunique()}" if 'category' in filtered_df.columns else "N/A")
    with col3:
        st.metric("Total Items", f"{len(filtered_df)}")

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

# --- ADD STATUS INFO ---
st.sidebar.markdown("---")
st.sidebar.markdown("<p class='section-header'>ℹ️ App Status</p>", unsafe_allow_html=True)
st.sidebar.info(f"Database location: {DB_PATH}\nMode: {'Development (Mock LLM)' if using_mock else 'Production'}")