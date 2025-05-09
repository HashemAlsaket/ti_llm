import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.express as px
import plotly.graph_objects as go
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_community.utilities import SQLDatabase
import logging

# --- SETUP ---
st.set_page_config(page_title="Tudor LLM Agent", layout="wide")

# --- CUSTOM THEME ---
# Define Tudor Investment theme colors based on logo
tudor_blue = "#0A50A1"  # Primary blue from Tudor logo
tudor_light_blue = "#3B7EC9"
tudor_dark_blue = "#063773"
background_color = "#FFFFFF"
accent_color = "#E5EBF3"

# Custom CSS with Tudor theme
st.markdown(f"""
<style>
    .stApp {{
        background-color: {background_color};
    }}
    .stButton>button {{
        background-color: {tudor_blue};
        color: white;
    }}
    .stButton>button:hover {{
        background-color: {tudor_light_blue};
    }}
    .st-cb, .st-bq, .st-an, .st-av, .st-at {{
        border-color: {tudor_blue};
    }}
    div[data-testid="stMetricValue"] {{
        color: {tudor_blue};
        font-weight: bold;
    }}
    div[data-testid="stExpander"] {{
        border-left-color: {tudor_blue} !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: {accent_color};
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {tudor_blue};
        color: white;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {tudor_dark_blue};
    }}
    .stSidebar .stButton>button {{
        width: 100%;
    }}
    a {{
        color: {tudor_blue};
    }}
</style>
""", unsafe_allow_html=True)

# Define the detailed schema as a constant at the top level for reuse
DETAILED_SCHEMA = """
Table: trades
Columns: id INTEGER, ticker TEXT, model_group TEXT, timestamp TIMESTAMP, position REAL, pnl REAL, 
alpha_score REAL, volatility REAL, sector TEXT, commodity_exposure REAL, interest_rate_sensitivity REAL
Example data: 
- (42, 'AAPL', 'Macro Alpha', '2025-02-26 00:00:00', 1733745.23, 25363.96, -0.498, 0.316, 'Energy', 0.791, 0.118)
- (51, 'MSFT', 'Tech Sector', '2025-01-10 00:00:00', 1896992.83, 198575.93, 1.814, 0.487, 'Technology', 0.237, 0.637)

Table: economic_indicators
Columns: id INTEGER, indicator_name TEXT, timestamp TIMESTAMP, value REAL, region TEXT, previous_value REAL
Example data:
- (1, 'GDP Growth', '2025-01-15 00:00:00', 2.8, 'US', 2.5)
- (8, 'Inflation Rate', '2025-02-15 00:00:00', 3.2, 'EU', 3.4)

Table: historical_trades
Columns: id INTEGER, ticker TEXT, trade_date TIMESTAMP, action TEXT, quantity INTEGER, price REAL, 
model_group TEXT, trade_id TEXT
Example data:
- (105, 'AAPL', '2025-01-23 00:00:00', 'BUY', 5000, 188.45, 'Macro Alpha', 'TRD-58291')
- (207, 'NVDA', '2025-02-14 00:00:00', 'SELL', 1200, 721.33, 'Tech Sector', 'TRD-83921')

Table: market_news
Columns: id INTEGER, title TEXT, summary TEXT, timestamp TIMESTAMP, source TEXT, url TEXT, 
tickers TEXT, sentiment REAL, relevance TEXT
Example data:
- (24, 'Fed Signals Rate Cuts', 'Federal Reserve hints at potential rate cuts in Q3', '2025-03-20 00:00:00', 'Bloomberg', 'http://example.com/news/24', 'SPY,QQQ,TLT', 0.75, 'High')
- (31, 'Tech Earnings Beat Expectations', 'Major tech companies report better than expected Q1 earnings', '2025-04-15 00:00:00', 'CNBC', 'http://example.com/news/31', 'AAPL,MSFT,GOOG', 0.82, 'High')

Table: simulated_stock_data
Columns: id INTEGER, ticker TEXT, timestamp TIMESTAMP, open REAL, high REAL, low REAL, close REAL, volume INTEGER
Example data:
- (1523, 'AAPL', '2025-02-15 00:00:00', 182.45, 184.95, 181.22, 184.37, 75231542)
- (2871, 'NVDA', '2025-03-22 00:00:00', 875.30, 915.75, 869.44, 908.88, 42567123)

Table: real_stock_data
Columns: id INTEGER, ticker TEXT, timestamp TIMESTAMP, open REAL, high REAL, low REAL, close REAL, 
volume INTEGER, last_refreshed TIMESTAMP
Example data:
- (4521, 'AAPL', '2025-04-01 00:00:00', 172.88, 174.30, 170.92, 173.05, 68254123, '2025-04-01 16:00:00')
- (5782, 'MSFT', '2025-03-15 00:00:00', 415.25, 419.88, 412.55, 418.52, 31254789, '2025-03-15 16:00:00')

Table: positions
Columns: timestamp TIMESTAMP, id INTEGER, position REAL, pnl REAL, alpha_score REAL, volatility REAL, 
sector TEXT, commodity_exposure REAL, interest_rate_sensitivity REAL
Example data:
- ('2025-01-07 00:00:00', 1, -216668.99, -115008.78, -2.01, 0.157, 'Energy', 0.839, 0.173)
- ('2025-02-06 00:00:00', 2, -1996884.94, 197274.05, -0.493, 0.347, 'Technology', 0.006, 0.691)
- ('2025-03-27 00:00:00', 3, -400556.11, -133667.02, 0.222, 0.253, 'Materials', 0.657, 0.273)
"""

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
        volatility REAL NOT NULL,
        sector TEXT,
        commodity_exposure REAL,
        interest_rate_sensitivity REAL
    )
    ''')
    
    # Create economic_indicators table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS economic_indicators (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        indicator_name TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        value REAL NOT NULL,
        region TEXT NOT NULL,
        previous_value REAL
    )
    ''')
    
    # Create historical_trades table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS historical_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        trade_date TIMESTAMP NOT NULL,
        action TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        price REAL NOT NULL,
        model_group TEXT,
        trade_id TEXT
    )
    ''')
    
    # Create market_news table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS market_news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        summary TEXT,
        timestamp TIMESTAMP NOT NULL,
        source TEXT,
        url TEXT,
        tickers TEXT,
        sentiment REAL,
        relevance TEXT
    )
    ''')
    
    # Create simulated_stock_data table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS simulated_stock_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume INTEGER
    )
    ''')
    
    # Create real_stock_data table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS real_stock_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume INTEGER,
        last_refreshed TIMESTAMP
    )
    ''')
    
    # Create positions table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS positions (
        timestamp TIMESTAMP NOT NULL,
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        position REAL NOT NULL,
        pnl REAL NOT NULL,
        alpha_score REAL NOT NULL,
        volatility REAL NOT NULL,
        sector TEXT,
        commodity_exposure REAL,
        interest_rate_sensitivity REAL
    )
    ''')
    
    conn.commit()
    conn.close()

def db_is_empty():
    """Check if database is empty"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check trades table
    cursor.execute("SELECT COUNT(*) FROM trades")
    trades_count = cursor.fetchone()[0]
    
    # Check economic_indicators table
    cursor.execute("SELECT COUNT(*) FROM economic_indicators")
    indicators_count = cursor.fetchone()[0]
    
    # Check historical_trades table
    cursor.execute("SELECT COUNT(*) FROM historical_trades")
    historical_trades_count = cursor.fetchone()[0]
    
    conn.close()
    return trades_count == 0 and indicators_count == 0 and historical_trades_count == 0

def load_data_to_db():
    """Generate mock data and load into SQLite database"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate trades data
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'XOM', 'CVX', 'BP', 'GOLD', 'NEM', 'RIO', 'VALE', 'USO', 'SILVER']
    model_groups = ['Macro Alpha', 'Q1 Equity', 'Commodities Signal', 'Rates Momentum', 'Tech Sector', 'Energy Focus', 'Mining Beta']
    sectors = ['Technology', 'Energy', 'Materials', 'Consumer Discretionary', 'Financial Services']
    
    trades_data = []
    for _ in range(500):
        ticker = np.random.choice(tickers)
        
        # Assign sectors based on ticker
        if ticker in ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA']:
            sector = 'Technology'
        elif ticker in ['XOM', 'CVX', 'BP', 'USO']:
            sector = 'Energy'
        elif ticker in ['GOLD', 'NEM', 'RIO', 'VALE', 'SILVER']:
            sector = 'Materials'
        else:
            sector = np.random.choice(sectors)
        
        # Determine commodity exposure based on sector
        if sector == 'Energy':
            commodity_exposure = np.random.uniform(0.6, 0.9)
        elif sector == 'Materials':
            commodity_exposure = np.random.uniform(0.5, 0.8)
        else:
            commodity_exposure = np.random.uniform(0, 0.3)
        
        # Determine interest rate sensitivity
        if sector == 'Financial Services':
            interest_rate_sensitivity = np.random.uniform(0.7, 0.95)
        elif sector == 'Technology':
            interest_rate_sensitivity = np.random.uniform(0.4, 0.7)
        else:
            interest_rate_sensitivity = np.random.uniform(0.1, 0.5)
        
        trades_data.append({
            "ticker": ticker,
            "model_group": np.random.choice(model_groups),
            "timestamp": datetime(2025, np.random.randint(1, 5), np.random.randint(1, 29)),
            "position": np.random.uniform(-2000000, 2000000),
            "pnl": np.random.uniform(-150000, 200000),
            "alpha_score": np.random.normal(0, 1),
            "volatility": np.random.uniform(0.1, 0.5),
            "sector": sector,
            "commodity_exposure": commodity_exposure,
            "interest_rate_sensitivity": interest_rate_sensitivity
        })
    
    # Generate economic indicators data
    indicators = ['GDP Growth', 'Inflation Rate', 'Unemployment', 'Interest Rate', 'Oil Price', 'Gold Price', 'Consumer Confidence']
    regions = ['US', 'EU', 'Asia', 'Global']
    
    economic_data = []
    for indicator in indicators:
        for region in regions:
            # Create time series of weekly values for the past 12 weeks
            for week in range(12):
                base_value = 0
                week_date = datetime.now() - timedelta(weeks=week)
                
                # Set base values for different indicators
                if indicator == 'GDP Growth':
                    base_value = np.random.uniform(2.0, 3.5)
                elif indicator == 'Inflation Rate':
                    base_value = np.random.uniform(2.5, 4.0)
                elif indicator == 'Unemployment':
                    base_value = np.random.uniform(3.5, 7.0)
                elif indicator == 'Interest Rate':
                    base_value = np.random.uniform(3.0, 5.0)
                elif indicator == 'Oil Price':
                    base_value = np.random.uniform(70, 90)
                elif indicator == 'Gold Price':
                    # Create a trend for gold prices
                    base_value = 1800 + (week * 10) + np.random.uniform(-20, 20)
                elif indicator == 'Consumer Confidence':
                    base_value = np.random.uniform(95, 110)
                
                # Add some random variation
                current_value = base_value + np.random.uniform(-0.5, 0.5)
                
                # Previous value (slightly different)
                previous_value = current_value + np.random.uniform(-0.3, 0.3)
                
                economic_data.append({
                    "indicator_name": indicator,
                    "timestamp": week_date,
                    "value": current_value,
                    "region": region,
                    "previous_value": previous_value
                })
    
    # Generate historical trades data
    historical_trades = []
    actions = ['BUY', 'SELL']
    
    for ticker in tickers:
        for _ in range(20):  # 20 trades per ticker
            trade_date = datetime(2025, np.random.randint(1, 5), np.random.randint(1, 29))
            price = 0
            
            # Set price ranges based on ticker
            if ticker == 'AAPL':
                price = np.random.uniform(160, 200)
            elif ticker == 'MSFT':
                price = np.random.uniform(320, 420)
            elif ticker == 'GOOG':
                price = np.random.uniform(130, 180)
            elif ticker == 'AMZN':
                price = np.random.uniform(150, 190)
            elif ticker == 'NVDA':
                price = np.random.uniform(700, 950)
            elif ticker in ['XOM', 'CVX', 'BP']:
                price = np.random.uniform(80, 120)
            elif ticker in ['GOLD', 'NEM', 'RIO', 'VALE']:
                price = np.random.uniform(30, 70)
            elif ticker == 'USO':
                price = np.random.uniform(60, 90)
            elif ticker == 'SILVER':
                price = np.random.uniform(20, 30)
            
            historical_trades.append({
                "ticker": ticker,
                "trade_date": trade_date,
                "action": np.random.choice(actions),
                "quantity": np.random.randint(100, 10000),
                "price": price,
                "model_group": np.random.choice(model_groups),
                "trade_id": f"TRD-{np.random.randint(10000, 99999)}"
            })
    
    # Generate positions data (based on the CSV data)
    positions_data = []
    for i in range(1, 110):  # Using the provided CSV data range
        timestamp = datetime(2025, np.random.randint(1, 5), np.random.randint(1, 29))
        sector = np.random.choice(['Technology', 'Energy', 'Materials'])
        
        # Determine commodity exposure based on sector
        if sector == 'Energy':
            commodity_exposure = np.random.uniform(0.6, 0.9)
        elif sector == 'Materials':
            commodity_exposure = np.random.uniform(0.5, 0.8)
        else:
            commodity_exposure = np.random.uniform(0, 0.3)
        
        # Determine interest rate sensitivity
        if sector == 'Financial Services':
            interest_rate_sensitivity = np.random.uniform(0.7, 0.95)
        elif sector == 'Technology':
            interest_rate_sensitivity = np.random.uniform(0.4, 0.7)
        else:
            interest_rate_sensitivity = np.random.uniform(0.1, 0.5)
        
        positions_data.append({
            "timestamp": timestamp,
            "id": i,
            "position": np.random.uniform(-2000000, 2000000),
            "pnl": np.random.uniform(-150000, 200000),
            "alpha_score": np.random.normal(0, 1),
            "volatility": np.random.uniform(0.1, 0.5),
            "sector": sector,
            "commodity_exposure": commodity_exposure,
            "interest_rate_sensitivity": interest_rate_sensitivity
        })
    
    # Generate simulated stock data for the past 12 weeks
    simulated_stock_data = []
    today = datetime.now()
    
    for ticker in tickers:
        # Set base price based on ticker
        if ticker == 'AAPL':
            base_price = 180
        elif ticker == 'MSFT':
            base_price = 350
        elif ticker == 'GOOG':
            base_price = 150
        elif ticker == 'AMZN':
            base_price = 170
        elif ticker == 'NVDA':
            base_price = 800
        elif ticker in ['XOM', 'CVX', 'BP']:
            base_price = 100
        elif ticker in ['GOLD', 'NEM', 'RIO', 'VALE']:
            base_price = 50
        elif ticker == 'USO':
            base_price = 70
        elif ticker == 'SILVER':
            base_price = 25
        else:
            base_price = 100
        
        # Generate daily data for the past 12 weeks
        for days_ago in range(84):  # 12 weeks * 7 days
            date = today - timedelta(days=days_ago)
            
            # Add some trend and seasonality
            trend = np.sin(days_ago / 30) * 10
            seasonality = np.sin(days_ago / 7) * 5
            
            # Apply specific trends for certain tickers
            if ticker == 'MSFT':
                # Microsoft has a rising trend
                trend = days_ago * 0.2
            elif ticker == 'GOOG':
                # Google has a cyclical pattern
                trend = np.sin(days_ago / 20) * 20
            elif ticker == 'GOLD':
                # Gold has a generally rising trend
                trend = days_ago * 0.1 + np.sin(days_ago / 15) * 5
            
            daily_volatility = np.random.uniform(0.01, 0.03)
            price_noise = np.random.normal(0, daily_volatility * base_price)
            
            adjusted_price = base_price + trend + seasonality + price_noise
            
            # Calculate day's OHLC
            open_price = adjusted_price
            high_price = adjusted_price * (1 + np.random.uniform(0, 0.02))
            low_price = adjusted_price * (1 - np.random.uniform(0, 0.02))
            close_price = adjusted_price * (1 + np.random.normal(0, 0.01))
            
            # Ensure high is highest and low is lowest
            high_price = max(open_price, close_price, high_price)
            low_price = min(open_price, close_price, low_price)
            
            # Volume varies by day of week (higher on Mon/Fri)
            day_of_week = date.weekday()
            volume_factor = 1.2 if day_of_week in [0, 4] else 1.0
            volume = int(np.random.uniform(500000, 5000000) * volume_factor)
            
            # Skip weekends for realism
            if day_of_week < 5:  # Monday to Friday only
                simulated_stock_data.append({
                    "ticker": ticker,
                    "timestamp": date,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume
                })
    
    # Insert data into database
    conn = sqlite3.connect(DB_PATH)
    
    trades_df = pd.DataFrame(trades_data)
    trades_df.to_sql('trades', conn, if_exists='append', index=False)
    
    economic_df = pd.DataFrame(economic_data)
    economic_df.to_sql('economic_indicators', conn, if_exists='append', index=False)
    
    historical_df = pd.DataFrame(historical_trades)
    historical_df.to_sql('historical_trades', conn, if_exists='append', index=False)
    
    positions_df = pd.DataFrame(positions_data)
    positions_df.to_sql('positions', conn, if_exists='append', index=False)
    
    simulated_stock_df = pd.DataFrame(simulated_stock_data)
    simulated_stock_df.to_sql('simulated_stock_data', conn, if_exists='append', index=False)
    
    # Copy simulated data to real_stock_data as well
    real_stock_data = []
    for _, row in simulated_stock_df.iterrows():
        real_stock_data.append({
            "ticker": row['ticker'],
            "timestamp": row['timestamp'],
            "open": row['open'],
            "high": row['high'],
            "low": row['low'],
            "close": row['close'],
            "volume": row['volume'],
            "last_refreshed": datetime.now()
        })
    
    real_stock_df = pd.DataFrame(real_stock_data)
    real_stock_df.to_sql('real_stock_data', conn, if_exists='append', index=False)
    
    conn.commit()
    conn.close()
    
    return trades_df

def get_data_from_db(ticker="All", model_group="All"):
    """Fetch data from SQLite database with optional filters using LLM-generated SQL"""
    conn = sqlite3.connect(DB_PATH)
    
    # Build natural language description of the query based on filters
    query_description = "Get all trades"
    filter_descriptions = []
    
    if ticker != "All":
        filter_descriptions.append(f"ticker is {ticker}")
    if model_group != "All":
        filter_descriptions.append(f"model group is {model_group}")
    
    if filter_descriptions:
        query_description += " where " + " and ".join(filter_descriptions)
    
    try:
        # Create a custom prompt template that includes the detailed schema
        sql_generation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "You are an expert SQL query generator for a financial analytics system. "
                "Generate a precise, efficient SQL query to answer the user's question based on the database schema provided. "
                "The query should be well-optimized and follow best practices. "
                "All SQL queries must conform to SQLite3 syntax. "
                "Return ONLY the SQL query, with no additional explanation."
                "If you are asked about making a prediction, performing a regression, future information etc., just pull the relevant past data (DO NOT ATTEMPT TO PULL FUTURE DATA), the CALCULATION AGENT will perform the calculation on that data"
            )),
            ("human", """
            Database Schema:
            {schema}
            
            User Question: {question}
            
            Please generate a SQL query to answer this question:
            """)
        ])
        
        # Generate SQL query using the custom prompt
        chain = sql_generation_prompt | langchain_llm | StrOutputParser()
        sql_query = chain.invoke({
            "schema": DETAILED_SCHEMA,
            "question": query_description
        })
        
        # Execute the query
        df = pd.read_sql(sql_query, conn)
        
    except Exception as e:
        # Log the error
        logging.error(f"LLM query generation failed: {str(e)}. Using fallback query.")
        
        # Fallback to manual query construction if LLM query fails
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

def execute_sql_query(prompt, thinking_container):
    """Use LangChain to convert natural language to SQL and execute, with thinking log"""
    # Custom styled sections with Tudor theme
    thinking_container.markdown("""
    <div style="border-left: 4px solid #0A50A1; padding-left: 20px; margin-bottom: 15px;">
        <h4 style="color: #0A50A1;">🧠 Agent Thinking: Converting your question to SQL</h4>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(0.5)
    
    thinking_container.markdown("""
    <div style="border-left: 4px solid #0A50A1; padding-left: 20px; margin-bottom: 15px;">
        <h4 style="color: #0A50A1;">📋 Database Schema</h4>
    </div>
    """, unsafe_allow_html=True)
    thinking_container.code(DETAILED_SCHEMA, language="sql")
    
    # Create a custom prompt template that includes the detailed schema
    sql_generation_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are an expert SQL query generator for a financial analytics system. "
            "Generate a precise, efficient SQL query to answer the user's question based on the database schema provided. "
            "The query should be well-optimized and follow best practices. "
            "All SQL queries must conform to SQLite3 syntax. Do not use MySQL, PostgreSQL, or other dialect-specific functions. "
            "Use SQLite functions such as date(), strftime(), and CURRENT_TIMESTAMP for date manipulation. "
            "Return ONLY the SQL query, with no additional explanation."
                "If you are asked about making a prediction, performing a regression, future information etc., just pull the relevant past data (DO NOT ATTEMPT TO PULL FUTURE DATA), the CALCULATION AGENT will perform the calculation on that data"
        )),
        ("human", """
        Database Schema:
        {schema}
        
        User Question: {question}
        
        Please generate a SQL query to answer this question:
        """)
    ])
    
    try:
        # Generate SQL query using the custom prompt
        thinking_container.markdown("""
        <div style="border-left: 4px solid #0A50A1; padding-left: 20px; margin-bottom: 15px;">
            <h4 style="color: #0A50A1;">🔍 Generating SQL Query</h4>
        </div>
        """, unsafe_allow_html=True)
        
        chain = sql_generation_prompt | langchain_llm | StrOutputParser()
        sql_query = chain.invoke({
            "schema": DETAILED_SCHEMA,
            "question": prompt
        })
        
        thinking_container.markdown("""
        <div style="border-left: 4px solid #0A50A1; padding-left: 20px; margin-bottom: 15px;">
            <h4 style="color: #0A50A1;">📝 Generated SQL</h4>
        </div>
        """, unsafe_allow_html=True)
        thinking_container.code(sql_query, language="sql")
        
        # Execute the query
        thinking_container.markdown("""
        <div style="border-left: 4px solid #0A50A1; padding-left: 20px; margin-bottom: 15px;">
            <h4 style="color: #0A50A1;">⚙️ Executing SQL Query</h4>
        </div>
        """, unsafe_allow_html=True)
        conn = sqlite3.connect(DB_PATH)
        results = pd.read_sql(sql_query, conn)
        conn.close()
        
        thinking_container.markdown("""
        <div style="border-left: 4px solid #0A50A1; padding-left: 20px; margin-bottom: 15px;">
            <h4 style="color: #0A50A1;">📊 Query Results</h4>
        </div>
        """, unsafe_allow_html=True)
        thinking_container.dataframe(results.head(10) if len(results) > 10 else results)
        
        # Get the LLM to explain the results
        thinking_container.markdown("""
        <div style="border-left: 4px solid #0A50A1; padding-left: 20px; margin-bottom: 15px;">
            <h4 style="color: #0A50A1;">🧩 Interpreting Results</h4>
        </div>
        """, unsafe_allow_html=True)
        
        explain_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "You are a financial analyst assistant for Tudor Investments. "
                "Explain the following SQL query and its results in a clear, concise way. "
                "Focus on the business insights and implications for investment decisions. "
                "If relevant, mention potential trading strategies based on the data."
            )),
            ("human", """
            Database Schema:
            {schema}
            
            SQL Query: {query}
            
            Results: {results}
            
            User question: {question}
            
            Please provide a detailed analysis of these results:
            """)
        ])
        
        chain = explain_prompt | langchain_llm | StrOutputParser()
        explanation = chain.invoke({
            "schema": DETAILED_SCHEMA,
            "query": sql_query,
            "results": results.to_string(),
            "question": prompt
        })
        
        thinking_container.markdown("""
        <div style="border-left: 4px solid #0A50A1; padding-left: 20px; margin-bottom: 15px;">
            <h4 style="color: #0A50A1;">✅ Analysis Complete: Preparing final response</h4>
        </div>
        """, unsafe_allow_html=True)
        
        return {
            "sql": sql_query,
            "results": results,
            "explanation": explanation
        }
    except Exception as e:
        thinking_container.error(f"❌ **Error:** {str(e)}")
        return {
            "error": str(e),
            "sql": "Error generating or executing SQL"
        }

# --- LOGIN FLOW ---
def login():
    st.session_state["authenticated"] = False

    with st.form("Login"):
        st.image("https://i.ibb.co/QKhpJDL/tudor-logo.png", width=100)
        st.write("🔐 Please log in to continue")
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            # Use appropriate credential checking based on your setup
            # This is a simplified example
            try:
                if user == st.secrets["credentials"]["username"] and pw == st.secrets["credentials"]["password"]:
                    st.session_state["authenticated"] = True
                else:
                    st.error("Invalid credentials")
            except Exception:
                # Fallback for local development without secrets
                if user == "admin" and pw == "password":
                    st.session_state["authenticated"] = True
                else:
                    st.error("Invalid credentials")

# Check authentication state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
    st.stop()

# --- HEADER WITH LOGO ---
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://i.ibb.co/QKhpJDL/tudor-logo.png", width=100)
with col2:
    st.title("📊 TI LLM Agent Prototype")

# --- INITIALIZE DATABASE ---
init_db()

# Load sample data if database is empty
if db_is_empty():
    with st.spinner("Loading initial data..."):
        load_data_to_db()
        st.success("Sample data loaded successfully!")

# --- INITIALIZE LANGCHAIN LLM ---
try:
    langchain_llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4")
except Exception:
    # Fallback for local development
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if openai_api_key:
        langchain_llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4")
    else:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")
        st.stop()

# --- SETUP LANGCHAIN SQL DATABASE ---
db_url = f"sqlite:///{DB_PATH}"
db = SQLDatabase.from_uri(db_url)

# --- SIDEBAR FILTERS ---
st.sidebar.image("https://i.ibb.co/QKhpJDL/tudor-logo.png", width=80)
st.sidebar.header("🔍 Filter Trades")
ticker_options = ["All"]
model_options = ["All"]

# Get unique values from the database
conn = sqlite3.connect(DB_PATH)
ticker_options += [row[0] for row in conn.execute("SELECT DISTINCT ticker FROM trades ORDER BY ticker").fetchall()]
model_options += [row[0] for row in conn.execute("SELECT DISTINCT model_group FROM trades ORDER BY model_group").fetchall()]
conn.close()

ticker = st.sidebar.selectbox("Select Ticker", ticker_options)
model = st.sidebar.selectbox("Select Model Group", model_options)

# --- GET FILTERED DATA ---
filtered_df = get_data_from_db(ticker, model)

# --- DISPLAY DATA ---
st.markdown("""
<div style='background-color: #E5EBF3; padding: 10px; border-radius: 5px; border-left: 5px solid #0A50A1;'>
    <h3 style='color: #0A50A1; margin: 0;'>🔁 Trade Data</h3>
</div>
""", unsafe_allow_html=True)
st.dataframe(filtered_df, use_container_width=True)

# --- SUMMARY METRICS ---
total_pnl = filtered_df['pnl'].sum() if 'pnl' in filtered_df.columns else 0
total_position = filtered_df['position'].sum() if 'position' in filtered_df.columns else 0
avg_alpha = filtered_df['alpha_score'].mean() if 'alpha_score' in filtered_df.columns else 0

# Define custom metric styles
metric_style = """
<style>
    div[data-testid="metric-container"] {
        background-color: #E5EBF3;
        border: 1px solid #0A50A1;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="metric-container"] > div {
        background-color: transparent;
    }
    div[data-testid="metric-container"] label {
        color: #063773;
        font-weight: 600;
    }
    div[data-testid="metric-container"] .stMetricValue {
        color: #0A50A1;
    }
</style>
"""
st.markdown(metric_style, unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("Total PnL", f"${total_pnl:,.0f}")
col2.metric("Net Position", f"${total_position:,.0f}")
col3.metric("Avg Alpha Score", f"{avg_alpha:.2f}")

# --- TEST PROMPTS ---
test_prompts = [
    "Which model group has the highest total PnL?",
    "What is the relationship between alpha score and volatility?",
    "Show me the performance of technology stocks compared to energy stocks",
    "Which ticker has the most negative position?",
    "What's the average PnL for trades in the Materials sector?",
    "How does commodity exposure correlate with PnL across different sectors?",
    "What was the trend of oil prices in the first quarter of 2025?",
    "Which region had the highest interest rates in March 2025?",
    "What is the distribution of trade sizes for NVDA?",
    "How have gold prices changed month-over-month in 2025?",
    "Compare the performance of the Macro Alpha and Tech Sector model groups",
    "What's the total position value by sector?",
    "Which model has the most consistent alpha score?",
    "Is there a correlation between interest rates and financial sector performance?",
    "What was the average buy price for Apple stock in Q1 2025?",
    "How does consumer confidence relate to position sizes in Consumer Discretionary stocks?",
    "Which commodity-related tickers have the highest volatility?",
    "What is the trend in inflation rates across different regions?",
    "Show me the relationship between GDP growth and stock performance by sector",
    "What was the largest single trade by value in the historical trades data?"
]

# --- LLM QUERY INTERFACE ---
st.markdown("""
<div style='background-color: #E5EBF3; padding: 10px; border-radius: 5px; border-left: 5px solid #0A50A1;'>
    <h3 style='color: #0A50A1; margin: 0;'>🤖 Ask a Question</h3>
</div>
""", unsafe_allow_html=True)

# Add dropdown for test prompts
use_test_prompt = st.checkbox("Use a test prompt", value=False)
if use_test_prompt:
    selected_prompt = st.selectbox("Select a test prompt", test_prompts)
    user_prompt = selected_prompt
else:
    user_prompt = st.text_area("What would you like to know about this trading data?", 
                            placeholder="Example: Which model group has the highest PnL?")

# Filter context prompt
filtered_context = f"Current filter: Ticker={ticker}, Model Group={model}"

if st.button("Ask the TI LLM Agent", help="Click to analyze your question with SQL", use_container_width=True):
    if user_prompt:
        # Create thinking log container
        thinking_container = st.expander("💡 Agent Thinking Process", expanded=True)
        
        with st.spinner("Processing your question..."):
            # Add filter context to prompt if filters are applied
            enhanced_prompt = user_prompt
            if ticker != "All" or model != "All":
                thinking_container.write(f"📌 **Applied Filters:** {filtered_context}")
                enhanced_prompt = f"{user_prompt} (Context: {filtered_context})"
            
            # Execute SQL query and get results
            result = execute_sql_query(enhanced_prompt, thinking_container)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                # Check if this is a visualization query with a chart
                if "chart" in result:
                    # Display chart in main content area
                    display_chart_in_main_area(result["chart"], st)
                
                # Display the explanation
                st.markdown(f"""
                <div style="background-color: #E5EBF3; border: 2px solid #0A50A1; 
                     padding: 20px; border-radius: 5px; margin-bottom: 20px;">
                    <h3 style="color: #0A50A1; margin-top: 0;">Analysis Results</h3>
                    <div style="color: #000;">{result["explanation"]}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # For standard SQL queries, show the SQL and raw results
                if "sql" in result and result["sql"] not in ["specialized_visualization_query", "specialized_prediction_query", "specialized_historical_query"]:
                    col1, col2 = st.columns(2)
                    with col1:
                        with st.expander("View SQL Query"):
                            st.code(result["sql"], language="sql")
                    
                    with col2:
                        with st.expander("View Raw Results"):
                            st.dataframe(result["results"])
    else:
        st.warning("Please enter a question.")

# --- ADD DATA UPLOAD FUNCTIONALITY ---
st.markdown("""
<div style='background-color: #E5EBF3; padding: 10px; border-radius: 5px; border-left: 5px solid #0A50A1;'>
    <h3 style='color: #0A50A1; margin: 0;'>📤 Upload Additional Data</h3>
</div>
""", unsafe_allow_html=True)

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
            
            if st.button("Confirm Upload to Database", use_container_width=True):
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