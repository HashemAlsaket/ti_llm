import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os
import time
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
    
    # Create market_news table
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
    
    # Create simulated_stock_data table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS simulated_stock_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER
    )
    ''')
    
    conn.commit()
    conn.close()

def db_has_data(table_name):
    """Check if a specific table has data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0

def load_data_to_db():
    """Generate mock data and load into SQLite database"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate trades data
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'XOM', 'CVX', 'BP', 'GOLD', 'NEM', 'RIO', 'VALE', 'USO', 'GLD', 'SLV']
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
        elif ticker in ['GOLD', 'NEM', 'RIO', 'VALE', 'GLD', 'SLV']:
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
            "timestamp": datetime(2024, np.random.randint(1, 5), np.random.randint(1, 29)),
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
            # Create time series of monthly values
            for month in range(1, 5):
                base_value = 0
                
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
                    base_value = np.random.uniform(1800, 2100)
                elif indicator == 'Consumer Confidence':
                    base_value = np.random.uniform(95, 110)
                
                # Add some random variation
                current_value = base_value + np.random.uniform(-0.5, 0.5)
                
                # Previous value (slightly different)
                previous_value = current_value + np.random.uniform(-0.3, 0.3)
                
                economic_data.append({
                    "indicator_name": indicator,
                    "timestamp": datetime(2024, month, 15),
                    "value": current_value,
                    "region": region,
                    "previous_value": previous_value
                })
    
    # Generate historical trades data
    historical_trades = []
    actions = ['BUY', 'SELL']
    
    for ticker in tickers:
        for _ in range(20):  # 20 trades per ticker
            trade_date = datetime(2024, np.random.randint(1, 5), np.random.randint(1, 29))
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
            elif ticker == 'GLD':
                price = np.random.uniform(180, 210)
            elif ticker == 'SLV':
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
    
    # Generate synthetic news data based on recent market trends
    news_data = [
        {
            "title": "Tariff Concerns Rattle Markets as S&P 500 Enters Correction Territory",
            "summary": "U.S. stocks entered correction territory following President Trump's announcement of sweeping new tariffs on imported goods, with small-cap stocks already in a bear market.",
            "timestamp": datetime(2025, 4, 5),
            "source": "US Bank",
            "url": "https://www.usbank.com/example",
            "tickers": "SPY,QQQ,IWM",
            "sentiment": -0.7,
            "relevance": "Major Market Movement"
        },
        {
            "title": "IMF Raises US Recession Risk to 40% Due to Trade Tensions",
            "summary": "The International Monetary Fund has increased the probability of a US recession to 40%, citing tariff policies as pushing the global economy towards a significant slowdown.",
            "timestamp": datetime(2025, 4, 28),
            "source": "World Economic Forum",
            "url": "https://www.weforum.org/example",
            "tickers": "SPY,DIA,EEM",
            "sentiment": -0.5,
            "relevance": "Economic Outlook"
        },
        {
            "title": "Fed Expected to Maintain Rates Amid Inflation Concerns",
            "summary": "The Federal Reserve is likely to maintain current interest rates as it balances inflation concerns with economic growth prospects in an uncertain trade environment.",
            "timestamp": datetime(2025, 5, 1),
            "source": "Edward Jones",
            "url": "https://www.edwardjones.com/example",
            "tickers": "TLT,IEF,BND",
            "sentiment": 0.1,
            "relevance": "Monetary Policy"
        },
        {
            "title": "Tech Stocks Hit Hardest as Trade Tensions Escalate",
            "summary": "Technology and consumer discretionary companies, many of which rely on overseas suppliers, have been particularly affected by the recent market selloff.",
            "timestamp": datetime(2025, 4, 4),
            "source": "Schwab Market Perspective",
            "url": "https://www.schwab.com/example",
            "tickers": "XLK,QQQ,AAPL,MSFT",
            "sentiment": -0.6,
            "relevance": "Sector Impact"
        },
        {
            "title": "Global Growth Projected at 3.3% for 2025 Despite Trade Concerns",
            "summary": "The IMF maintains its global growth projection at 3.3% for 2025, with upward revisions for the United States offsetting downward adjustments elsewhere.",
            "timestamp": datetime(2025, 4, 15),
            "source": "IMF World Economic Outlook",
            "url": "https://www.imf.org/example",
            "tickers": "ACWI,EFA,SPY",
            "sentiment": 0.3,
            "relevance": "Economic Outlook"
        },
        {
            "title": "Cybersecurity Concerns Intensify for Financial Services in 2025",
            "summary": "Financial institutions face increased pressure to ensure systems meet evolving industry standards as cybersecurity threats intensify.",
            "timestamp": datetime(2025, 4, 18),
            "source": "Genesis Global",
            "url": "https://genesis.global/example",
            "tickers": "CIBR,HACK,FIN",
            "sentiment": -0.2,
            "relevance": "Industry Trend"
        },
        {
            "title": "Investors Pull Record Amount from ESG Funds in Q1 2025",
            "summary": "Sustainable funds saw record withdrawals in Q1, with US investors reducing exposure for the tenth consecutive quarter and Europeans becoming net sellers for the first time since 2018.",
            "timestamp": datetime(2025, 4, 25),
            "source": "World Economic Forum",
            "url": "https://www.weforum.org/example",
            "tickers": "ESGU,ESGD,ESGE",
            "sentiment": -0.4,
            "relevance": "Investment Trend"
        },
        {
            "title": "Oil Prices Surge on Supply Chain Disruptions and Geopolitical Tensions",
            "summary": "Crude oil prices have risen sharply due to global supply chain disruptions and escalating geopolitical tensions, affecting energy sector stocks.",
            "timestamp": datetime(2025, 5, 5),
            "source": "Financial Market News",
            "url": "https://example.com/news",
            "tickers": "XOM,CVX,USO,XLE",
            "sentiment": 0.6,
            "relevance": "Commodity Impact"
        },
        {
            "title": "Bitcoin Surpasses $100,000 as Cryptocurrency Adoption Accelerates",
            "summary": "Bitcoin has reached a new all-time high above $100,000 as institutional adoption of cryptocurrencies continues to increase in 2025.",
            "timestamp": datetime(2025, 5, 7),
            "source": "Crypto Market News",
            "url": "https://example.com/crypto",
            "tickers": "COIN,BTCUSD,GBTC",
            "sentiment": 0.8,
            "relevance": "Cryptocurrency"
        },
        {
            "title": "US Cargo Shipments Plummet by Up to 60% Since Early April",
            "summary": "Major retailers are warning of potential empty shelves and higher prices by mid-May due to a sharp decline in cargo shipments, with logistics and retail sectors facing possible layoffs.",
            "timestamp": datetime(2025, 4, 29),
            "source": "Supply Chain Report",
            "url": "https://example.com/supply",
            "tickers": "AMZN,WMT,TGT,UPS,FDX",
            "sentiment": -0.7,
            "relevance": "Supply Chain"
        }
    ]
    
    # Generate simulated stock data
    simulated_stock_data = []
    
    # For each ticker, generate 60 days of data
    for ticker in tickers:
        # Set base price for each ticker
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
            base_price = 75
        elif ticker == 'GLD':
            base_price = 195
        elif ticker == 'SLV':
            base_price = 25
        
        # Generate price trend
        trend = np.random.choice(['up', 'down', 'flat', 'volatile'])
        trend_strength = np.random.uniform(0.05, 0.2)  # 5-20% trend over the period
        
        # Generate data for each day
        for day in range(60):
            date = datetime(2025, 3, 1) + timedelta(days=day)
            
            # Apply trend
            if trend == 'up':
                multiplier = 1 + (day / 60) * trend_strength
            elif trend == 'down':
                multiplier = 1 - (day / 60) * trend_strength
            elif trend == 'flat':
                multiplier = 1 + np.random.uniform(-0.01, 0.01)
            else:  # volatile
                multiplier = 1 + np.random.uniform(-0.1, 0.1)
            
            # Apply tariff news impact for dates after April 2
            if date > datetime(2025, 4, 2) and ticker in ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA']:
                tariff_impact = 0.9  # 10% drop for tech stocks
            elif date > datetime(2025, 4, 2) and ticker in ['XOM', 'CVX', 'BP']:
                tariff_impact = 0.95  # 5% drop for energy stocks
            else:
                tariff_impact = 1.0
            
            # Calculate daily price
            daily_price = base_price * multiplier * tariff_impact
            
            # Daily volatility
            volatility = 0.015  # 1.5% average daily move
            
            # Calculate open, high, low, close
            open_price = daily_price * (1 + np.random.uniform(-volatility/2, volatility/2))
            close_price = daily_price * (1 + np.random.uniform(-volatility/2, volatility/2))
            high_price = max(open_price, close_price) * (1 + np.random.uniform(0, volatility))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0, volatility))
            
            # Volume (higher on volatile days)
            base_volume = 1000000  # Base volume in shares
            if abs(open_price - close_price) / open_price > volatility:
                volume_multiplier = np.random.uniform(1.2, 2.0)  # Higher volume on volatile days
            else:
                volume_multiplier = np.random.uniform(0.8, 1.2)
            
            volume = int(base_volume * volume_multiplier)
            
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
    
    news_df = pd.DataFrame(news_data)
    news_df.to_sql('market_news', conn, if_exists='append', index=False)
    
    stock_df = pd.DataFrame(simulated_stock_data)
    stock_df.to_sql('simulated_stock_data', conn, if_exists='append', index=False)
    
    conn.commit()
    conn.close()
    
    return trades_df

def check_and_load_additional_data():
    """Check if additional data tables have data, and if not, load them"""
    is_news_empty = not db_has_data('market_news')
    is_stock_empty = not db_has_data('simulated_stock_data')
    
    if is_news_empty or is_stock_empty:
        load_data_to_db()
        return True
    return False

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

def generate_market_insights():
    """Generate random daily market insights based on simulated data"""
    conn = sqlite3.connect(DB_PATH)
    
    # Get most recent stock data
    recent_stock_df = pd.read_sql(
        "SELECT ticker, close, timestamp FROM simulated_stock_data " +
        "WHERE timestamp = (SELECT MAX(timestamp) FROM simulated_stock_data)",
        conn
    )
    
    # Get previous day's close
    prev_day_stock_df = pd.read_sql(
        "SELECT ticker, close, timestamp FROM simulated_stock_data " +
        "WHERE timestamp = (SELECT MAX(timestamp) FROM simulated_stock_data WHERE timestamp < (SELECT MAX(timestamp) FROM simulated_stock_data))",
        conn
    )
    
    # Get latest news sentiment
    news_df = pd.read_sql(
        "SELECT sentiment FROM market_news ORDER BY timestamp DESC LIMIT 5",
        conn
    )
    
    conn.close()
    
    # Merge to calculate daily returns
    stock_df = recent_stock_df.merge(
        prev_day_stock_df,
        on='ticker',
        suffixes=('', '_prev')
    )
    stock_df['daily_return'] = (stock_df['close'] - stock_df['close_prev']) / stock_df['close_prev'] * 100
    
    # Calculate market summary
    top_performer = stock_df.loc[stock_df['daily_return'].idxmax()]
    worst_performer = stock_df.loc[stock_df['daily_return'].idxmin()]
    avg_return = stock_df['daily_return'].mean()
    
    # Calculate average sentiment
    avg_sentiment = news_df['sentiment'].mean() if not news_df.empty else 0
    
    # Generate market insight text
    if avg_return > 1.0:
        market_state = "strongly bullish"
    elif avg_return > 0.2:
        market_state = "mildly bullish"
    elif avg_return > -0.2:
        market_state = "relatively flat"
    elif avg_return > -1.0:
        market_state = "mildly bearish"
    else:
        market_state = "strongly bearish"
    
    # Generate sentiment text
    if avg_sentiment > 0.3:
        sentiment_text = "positive"
    elif avg_sentiment > -0.3:
        sentiment_text = "neutral"
    else:
        sentiment_text = "negative"
    
    date_str = recent_stock_df['timestamp'].iloc[0].strftime('%Y-%m-%d') if not recent_stock_df.empty else "today"
    
    insight_text = f"Market Insight for {date_str}: The market is {market_state} with an average return of {avg_return:.2f}%. "
    insight_text += f"The top performer is {top_performer['ticker']} (+{top_performer['daily_return']:.2f}%), "
    insight_text += f"while {worst_performer['ticker']} is the worst performer ({worst_performer['daily_return']:.2f}%). "
    insight_text += f"News sentiment is generally {sentiment_text}."
    
    return insight_text, top_performer['ticker'], worst_performer['ticker'], avg_return

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

# --- INITIALIZE DATABASE ---
init_db()

# Check if trades data exists
if not db_has_data('trades'):
    with st.spinner("Loading initial trades data..."):
        load_data_to_db()
        st.success("Sample data loaded successfully!")
else:
    # If trades exist but other data doesn't, load additional data
    with st.spinner("Checking for additional data..."):
        if check_and_load_additional_data():
            st.success("Additional market data loaded successfully!")

# --- INITIALIZE CLIENTS ---
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
langchain_llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4")

# --- SETUP LANGCHAIN SQL DATABASE ---
db_url = f"sqlite:///{DB_PATH}"
db = SQLDatabase.from_uri(db_url)

# --- SIDEBAR FILTERS ---
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

# --- QUERY APPROACH SELECTION ---
query_method = st.sidebar.radio(
    "LLM Query Method",
    ["Simple (Direct)", "SQL-Based", "Enhanced SQL"],
    help="Choose how the LLM will process your query"
)

# --- LLM QUERY FUNCTIONS WITH THINKING LOG ---
def ask_direct_llm(prompt, context, thinking_container):
    """Legacy direct LLM query method with thinking log"""
    thinking_container.write("🧠 **Agent Thinking:** Preparing to analyze data directly...")
    time.sleep(0.5)
    
    thinking_container.write("📊 **Data Sample:**")
    thinking_container.json(context[:500] + "..." if len(context) > 500 else context)
    time.sleep(1)
    
    system_prompt = (
        "You are a financial analyst assistant for Tudor Investments. "
        "Answer questions using the following data context."
    )
    
    thinking_container.write("🤔 **Processing Query:** Analyzing data to find patterns and insights...")
    time.sleep(1)

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Data context: {context}\n\nUser question: {prompt}"},
        ],
        temperature=0.2,
        max_tokens=500
    )
    
    thinking_container.write("✅ **Completed Analysis:** Ready to provide insights")
    return response.choices[0].message.content

def ask_sql_llm(prompt, thinking_container):
    """Use LangChain to convert natural language to SQL and execute, with thinking log"""
    # Get DB schema
    schema = get_db_schema()
    
    thinking_container.write("🧠 **Agent Thinking:** Converting your question to SQL...")
    time.sleep(0.5)
    
    thinking_container.write("📋 **Database Schema:**")
    thinking_container.code(schema, language="sql")
    time.sleep(1)
    
    # Create a chain that generates SQL
    sql_chain = create_sql_query_chain(
        langchain_llm,
        db,
        k=3  # Number of examples used for few-shot prompting
    )
    
    try:
        # Generate SQL query
        thinking_container.write("🔍 **Generating SQL Query...**")
        sql_query = sql_chain.invoke({"question": prompt})
        
        thinking_container.write("📝 **Generated SQL:**")
        thinking_container.code(sql_query, language="sql")
        time.sleep(1)
        
        # Execute the query
        thinking_container.write("⚙️ **Executing SQL Query...**")
        conn = sqlite3.connect(DB_PATH)
        results = pd.read_sql(sql_query, conn)
        conn.close()
        
        thinking_container.write("📊 **Query Results:**")
        thinking_container.dataframe(results.head(5) if len(results) > 5 else results)
        time.sleep(1)
        
        # Get the LLM to explain the results
        thinking_container.write("🧩 **Interpreting Results...**")
        
        explain_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "You are a financial analyst assistant for Tudor Investments. "
                "Explain the following SQL query and its results in a clear, concise way. "
                "Focus on the business insights and implications."
            )),
            ("human", "SQL Query: {query}\n\nResults: {results}\n\nUser question: {question}")
        ])
        
        chain = explain_prompt | langchain_llm | StrOutputParser()
        explanation = chain.invoke({
            "query": sql_query,
            "results": results.to_string(),
            "question": prompt
        })
        
        thinking_container.write("✅ **Analysis Complete:** Preparing final response")
        
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

def ask_enhanced_sql_llm(prompt, filtered_df, thinking_container):
    """Enhanced SQL-based approach with better context and thinking log"""
    thinking_container.write("🧠 **Agent Thinking:** Starting enhanced analysis...")
    time.sleep(0.5)
    
    # Get recent news for context
    thinking_container.write("📰 **Checking Recent Market News...**")
    conn = sqlite3.connect(DB_PATH)
    news_df = pd.read_sql("SELECT title, summary, timestamp, source, sentiment FROM market_news ORDER BY timestamp DESC LIMIT 5", conn)
    conn.close()
    
    thinking_container.write("📰 **Recent News Context:**")
    if not news_df.empty:
        thinking_container.dataframe(news_df)
    else:
        thinking_container.write("No recent news found in database.")
    time.sleep(1)
    
    # Try SQL approach
    thinking_container.write("⚡ **Phase 1:** Performing SQL query analysis")
    sql_response = ask_sql_llm(prompt, thinking_container)
    
    # Create a combined prompt with SQL results and filtered data context
    thinking_container.write("⚡ **Phase 2:** Enhancing with filtered data context")
    time.sleep(0.5)
    
    # Add news context to the prompt
    news_context = ""
    if not news_df.empty:
        news_context = "Recent Market News:\n"
        for i, row in news_df.iterrows():
            sentiment_label = "positive" if row['sentiment'] > 0.2 else "negative" if row['sentiment'] < -0.2 else "neutral"
            news_context += f"- {row['title']} ({row['source']}, {row['timestamp'].strftime('%Y-%m-%d')}, sentiment: {sentiment_label})\n"
    
    combined_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a sophisticated financial analyst assistant for Tudor Investments. "
            "Provide a comprehensive answer to the user's question using the SQL query results, "
            "the current filtered data context, and recent market news."
        )),
        ("human", """
        User question: {question}
        
        SQL Analysis:
        {sql_results}
        
        Current Filtered Data Summary:
        {filtered_data_summary}
        
        {news_context}
        """)
    ])
    
    # Create a summary of the filtered data
    filtered_summary = f"Current filter: Ticker={ticker}, Model Group={model}"
    thinking_container.write(f"📌 **Current Context:** {filtered_summary}")
    time.sleep(0.5)
    
    if "error" in sql_response:
        sql_results_text = f"SQL Error: {sql_response['error']}"
        thinking_container.error(f"❌ **SQL Error:** {sql_response['error']}")
    else:
        sql_results_text = f"SQL Query: {sql_response['sql']}\n\nResults Summary: {sql_response.get('explanation', 'No explanation available')}"
    
    thinking_container.write("🔄 **Synthesizing Complete Answer...**")
    
    chain = combined_prompt | langchain_llm | StrOutputParser()
    final_response = chain.invoke({
        "question": prompt,
        "sql_results": sql_results_text,
        "filtered_data_summary": filtered_summary,
        "news_context": news_context
    })
    
    thinking_container.write("✅ **Enhanced Analysis Complete:** Final response ready")
    
    return {
        "response": final_response,
        "sql": sql_response.get("sql", "No SQL query generated"),
        "sql_results": sql_response.get("results", pd.DataFrame()) if "error" not in sql_response else None
    }

# --- GET FILTERED DATA ---
filtered_df = get_data_from_db(ticker, model)

# --- Generate Daily Market Insight ---
insight_text, top_ticker, bottom_ticker, market_return = generate_market_insights()

# --- DISPLAY DATA ---
# Display market insight
st.info(insight_text)

# Create tabs
data_tab, news_tab, metrics_tab = st.tabs(["📊 Trade Data", "📰 Market News", "📈 Market Data"])

with data_tab:
    st.dataframe(filtered_df, use_container_width=True)

with news_tab:
    conn = sqlite3.connect(DB_PATH)
    news_df = pd.read_sql("SELECT title, summary, timestamp, source, sentiment FROM market_news ORDER BY timestamp DESC", conn)
    conn.close()
    
    if not news_df.empty:
        # Format sentiment values
        news_df['sentiment_formatted'] = news_df['sentiment'].apply(
            lambda x: "🟢 Positive" if x > 0.2 else "🔴 Negative" if x < -0.2 else "⚪ Neutral")
        
        st.dataframe(news_df[['title', 'summary', 'timestamp', 'source', 'sentiment_formatted']], 
                    use_container_width=True,
                    column_config={
                        "title": "Headline",
                        "summary": "Summary",
                        "timestamp": "Date",
                        "source": "Source",
                        "sentiment_formatted": "Sentiment"
                    })
    else:
        st.info("No news data available")

with metrics_tab:
    # Summary metrics
    total_pnl = filtered_df['pnl'].sum()
    total_position = filtered_df['position'].sum()
    avg_alpha = filtered_df['alpha_score'].mean()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total PnL", f"${total_pnl:,.0f}")
    col2.metric("Net Position", f"${total_position:,.0f}")
    col3.metric("Avg Alpha Score", f"{avg_alpha:.2f}")
    
    # Show simulated stock data
    st.subheader("Simulated Stock Data")
    
    # Allow user to select ticker for chart
    stock_ticker = st.selectbox("Select Stock for Chart", ticker_options[1:])  # Skip "All"
    
    if stock_ticker:
        conn = sqlite3.connect(DB_PATH)
        stock_df = pd.read_sql(f"SELECT * FROM simulated_stock_data WHERE ticker = '{stock_ticker}' ORDER BY timestamp", conn)
        conn.close()
        
        if not stock_df.empty:
            # Calculate daily returns
            stock_df['daily_return'] = stock_df['close'].pct_change() * 100
            
            st.subheader(f"{stock_ticker} Stock Performance")
            
            # Create two columns for price and volume
            price_col, volume_col = st.columns(2)
            
            with price_col:
                st.line_chart(stock_df.set_index('timestamp')['close'], use_container_width=True)
                st.caption("Closing Price")
            
            with volume_col:
                st.bar_chart(stock_df.set_index('timestamp')['volume'], use_container_width=True)
                st.caption("Volume")
            
            # Returns chart
            st.bar_chart(stock_df.set_index('timestamp')['daily_return'], use_container_width=True)
            st.caption("Daily Returns (%)")
            
            # Show recent data as table
            st.subheader("Recent Price Data")
            st.dataframe(stock_df.tail(10)[['timestamp', 'open', 'high', 'low', 'close', 'volume']])
        else:
            st.info(f"No data available for {stock_ticker}")

# --- TEST PROMPTS ---
test_prompts = [
    "Which model group has the highest total PnL?",
    "What is the relationship between alpha score and volatility?",
    "Show me the performance of technology stocks compared to energy stocks",
    "Which ticker has the most negative position?",
    "What's the average PnL for trades in the Materials sector?",
    "How does commodity exposure correlate with PnL across different sectors?",
    "What's the trend of oil prices based on economic indicators?",
    "Which region had the highest interest rates in March 2024?",
    "What is the distribution of trade sizes for NVDA?",
    "How have gold prices changed month-over-month in 2024?",
    "Compare the performance of the Macro Alpha and Tech Sector model groups",
    "What's the total position value by sector?",
    "Which model has the most consistent alpha score?",
    "Is there a correlation between interest rates and financial sector performance?",
    "What was the average buy price for Apple stock in Q1 2024?",
    "How does consumer confidence relate to position sizes in Consumer Discretionary stocks?",
    "Which commodity-related tickers have the highest volatility?",
    "What is the trend in inflation rates across different regions?",
    "Show me the relationship between GDP growth and stock performance by sector",
    "What was the largest single trade by value in the historical trades data?"
]

# Add news-related prompts
news_prompts = [
    "Summarize the recent market news sentiment and its potential impact on our positions",
    "What sectors are being mentioned most frequently in recent negative news?",
    "Is there a correlation between news sentiment and market performance for our tickers?",
    "Based on recent news, which of our positions might be most at risk?",
    "How might the recent tariff news impact our technology sector exposure?"
]
test_prompts.extend(news_prompts)

# --- LLM QUERY INTERFACE ---
st.subheader("🤖 Ask a Question")

# Add dropdown for test prompts
use_test_prompt = st.checkbox("Use a test prompt", value=False)
if use_test_prompt:
    selected_prompt = st.selectbox("Select a test prompt", test_prompts)
    user_prompt = selected_prompt
else:
    user_prompt = st.text_area("What would you like to know about this trading data?", 
                            placeholder="Example: Which model group has the highest PnL?")

# Create a placeholder for the agent thinking log
thinking_log = st.empty()

if st.button("Ask the TI LLM Agent"):
    if user_prompt:
        # Clear previous thinking log
        thinking_container = st.expander("💡 Agent Thinking Process", expanded=True)
        
        with st.spinner("Processing your question..."):
            if query_method == "Simple (Direct)":
                # Use the original direct approach
                sample_data = filtered_df.head(10).to_dict(orient="records")
                context_text = str(sample_data)
                answer = ask_direct_llm(user_prompt, context_text, thinking_container)
                
                st.success("LLM Response:")
                st.write(answer)
                
            elif query_method == "SQL-Based":
                # Use the SQL-based approach
                result = ask_sql_llm(user_prompt, thinking_container)
                
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
                result = ask_enhanced_sql_llm(user_prompt, filtered_df, thinking_container)
                
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
st.subheader("📤 Upload Additional Data")

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