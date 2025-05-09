import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os
import time
import requests
import json
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
    
    # Create real_stock_data table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS real_stock_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        last_refreshed TIMESTAMP
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
            "url": "https://www.usbank.com/investing/financial-perspectives/market-news/is-a-market-correction-coming.html",
            "tickers": "SPY,QQQ,IWM",
            "sentiment": -0.7,
            "relevance": "Major Market Movement"
        },
        {
            "title": "IMF Raises US Recession Risk to 40% Due to Trade Tensions",
            "summary": "The International Monetary Fund has increased the probability of a US recession to 40%, citing tariff policies as pushing the global economy towards a significant slowdown.",
            "timestamp": datetime(2025, 4, 28),
            "source": "World Economic Forum",
            "url": "https://www.weforum.org/stories/2025/04/imf-raises-us-recession-risk-and-other-finance-news-to-know/",
            "tickers": "SPY,DIA,EEM",
            "sentiment": -0.5,
            "relevance": "Economic Outlook"
        },
        {
            "title": "Fed Expected to Maintain Rates Amid Inflation Concerns",
            "summary": "The Federal Reserve is likely to maintain current interest rates as it balances inflation concerns with economic growth prospects in an uncertain trade environment.",
            "timestamp": datetime(2025, 5, 1),
            "source": "Edward Jones",
            "url": "https://www.edwardjones.com/us-en/market-news-insights/stock-market-news/stock-market-weekly-update",
            "tickers": "TLT,IEF,BND",
            "sentiment": 0.1,
            "relevance": "Monetary Policy"
        },
        {
            "title": "Tech Stocks Hit Hardest as Trade Tensions Escalate",
            "summary": "Technology and consumer discretionary companies, many of which rely on overseas suppliers, have been particularly affected by the recent market selloff.",
            "timestamp": datetime(2025, 4, 4),
            "source": "Schwab Market Perspective",
            "url": "https://www.schwab.com/learn/story/stock-market-outlook",
            "tickers": "XLK,QQQ,AAPL,MSFT",
            "sentiment": -0.6,
            "relevance": "Sector Impact"
        },
        {
            "title": "Global Growth Projected at 3.3% for 2025 Despite Trade Concerns",
            "summary": "The IMF maintains its global growth projection at 3.3% for 2025, with upward revisions for the United States offsetting downward adjustments elsewhere.",
            "timestamp": datetime(2025, 4, 15),
            "source": "IMF World Economic Outlook",
            "url": "https://www.imf.org/en/Publications/WEO",
            "tickers": "ACWI,EFA,SPY",
            "sentiment": 0.3,
            "relevance": "Economic Outlook"
        },
        {
            "title": "Cybersecurity Concerns Intensify for Financial Services in 2025",
            "summary": "Financial institutions face increased pressure to ensure systems meet evolving industry standards as cybersecurity threats intensify.",
            "timestamp": datetime(2025, 4, 18),
            "source": "Genesis Global",
            "url": "https://genesis.global/report/2025-trends-in-financial-markets/",
            "tickers": "CIBR,HACK,FIN",
            "sentiment": -0.2,
            "relevance": "Industry Trend"
        },
        {
            "title": "Investors Pull Record Amount from ESG Funds in Q1 2025",
            "summary": "Sustainable funds saw record withdrawals in Q1, with US investors reducing exposure for the tenth consecutive quarter and Europeans becoming net sellers for the first time since 2018.",
            "timestamp": datetime(2025, 4, 25),
            "source": "World Economic Forum",
            "url": "https://www.weforum.org/stories/2025/04/imf-raises-us-recession-risk-and-other-finance-news-to-know/",
            "tickers": "ESGU,ESGD,ESGE",
            "sentiment": -0.4,
            "relevance": "Investment Trend"
        },
        {
            "title": "Oil Prices Surge on Supply Chain Disruptions and Geopolitical Tensions",
            "summary": "Crude oil prices have risen sharply due to global supply chain disruptions and escalating geopolitical tensions, affecting energy sector stocks.",
            "timestamp": datetime(2025, 5, 5),
            "source": "Financial Market News",
            "url": "#",
            "tickers": "XOM,CVX,USO,XLE",
            "sentiment": 0.6,
            "relevance": "Commodity Impact"
        },
        {
            "title": "Bitcoin Surpasses $100,000 as Coinbase Acquires Deribit",
            "summary": "Bitcoin has jumped above $101,000 as cryptocurrency exchange Coinbase announced the acquisition of trading platform Deribit in a deal valued at $2.9 billion.",
            "timestamp": datetime(2025, 5, 7),
            "source": "Yahoo Finance",
            "url": "https://finance.yahoo.com",
            "tickers": "COIN,BTCUSD,GBTC",
            "sentiment": 0.8,
            "relevance": "Cryptocurrency"
        },
        {
            "title": "US Cargo Shipments Plummet by Up to 60% Since Early April",
            "summary": "Major retailers are warning of potential empty shelves and higher prices by mid-May due to a sharp decline in cargo shipments, with logistics and retail sectors facing possible layoffs.",
            "timestamp": datetime(2025, 4, 29),
            "source": "World Economic Forum",
            "url": "https://www.weforum.org/stories/2025/04/imf-raises-us-recession-risk-and-other-finance-news-to-know/",
            "tickers": "AMZN,WMT,TGT,UPS,FDX",
            "sentiment": -0.7,
            "relevance": "Supply Chain"
        }
    ]
    
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

def fetch_alpha_vantage_stock_data(ticker, api_key):
    """Fetch real stock data from Alpha Vantage API"""
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        
        if "Error Message" in data:
            return None, f"API Error: {data['Error Message']}"
        
        if "Time Series (Daily)" not in data:
            return None, f"No data found for ticker {ticker}"
        
        # Extract the time series data
        time_series = data["Time Series (Daily)"]
        last_refreshed = data["Meta Data"]["3. Last Refreshed"]
        
        # Convert to DataFrame
        df_data = []
        for date, values in time_series.items():
            df_data.append({
                "ticker": ticker,
                "timestamp": datetime.strptime(date, "%Y-%m-%d"),
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "volume": int(values["5. volume"]),
                "last_refreshed": datetime.strptime(last_refreshed, "%Y-%m-%d")
            })
        
        # Sort by date
        df = pd.DataFrame(df_data)
        df = df.sort_values(by="timestamp", ascending=False)
        
        return df, None
    except Exception as e:
        return None, f"Error fetching stock data: {str(e)}"

def fetch_alpha_vantage_news(api_key):
    """Fetch financial news from Alpha Vantage API"""
    try:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        
        if "Error Message" in data:
            return None, f"API Error: {data['Error Message']}"
        
        if "feed" not in data:
            return None, "No news data found"
        
        # Extract the news feed
        news_feed = data["feed"]
        
        # Convert to DataFrame
        df_data = []
        for news in news_feed:
            # Extract ticker symbols if available
            tickers = ""
            if "ticker_sentiment" in news and news["ticker_sentiment"]:
                tickers = ",".join([item["ticker"] for item in news["ticker_sentiment"]])
            
            # Calculate overall sentiment
            sentiment = 0.0
            if "overall_sentiment_score" in news:
                sentiment = float(news["overall_sentiment_score"])
            
            df_data.append({
                "title": news.get("title", "No Title"),
                "summary": news.get("summary", ""),
                "timestamp": datetime.strptime(news.get("time_published", "20250101T000000")[:8], "%Y%m%d"),
                "source": news.get("source", "Unknown"),
                "url": news.get("url", ""),
                "tickers": tickers,
                "sentiment": sentiment,
                "relevance": "API Source"
            })
        
        # Create DataFrame
        df = pd.DataFrame(df_data)
        
        return df, None
    except Exception as e:
        return None, f"Error fetching news data: {str(e)}"

def store_real_data_in_db(stock_df=None, news_df=None):
    """Store real data from APIs in the database"""
    conn = sqlite3.connect(DB_PATH)
    
    try:
        if stock_df is not None and not stock_df.empty:
            stock_df.to_sql('real_stock_data', conn, if_exists='append', index=False)
        
        if news_df is not None and not news_df.empty:
            news_df.to_sql('market_news', conn, if_exists='append', index=False)
        
        conn.commit()
        success = True
        error = None
    except Exception as e:
        conn.rollback()
        success = False
        error = str(e)
    finally:
        conn.close()
    
    return success, error

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

# --- ALPHA VANTAGE API KEY ---
alpha_vantage_api_key = st.secrets.get("ALPHA_VANTAGE_API_KEY", "demo")

# --- FETCH REAL DATA SECTION ---
st.sidebar.header("📈 Fetch Real Data")

fetch_real_data = st.sidebar.checkbox("Fetch Real Market Data", value=False)

if fetch_real_data:
    with st.sidebar.expander("Alpha Vantage API Settings"):
        # Option to input API key directly
        use_custom_key = st.checkbox("Use Custom API Key", value=False)
        if use_custom_key:
            custom_api_key = st.text_input("Alpha Vantage API Key", "")
            if custom_api_key:
                alpha_vantage_api_key = custom_api_key
        
        fetch_stock_data = st.checkbox("Fetch Stock Data", value=True)
        fetch_news_data = st.checkbox("Fetch News Data", value=True)
        
        if fetch_stock_data:
            stock_ticker = st.text_input("Stock Ticker Symbol", "MSFT")
        
        if st.button("Fetch Data"):
            with st.spinner("Fetching data from Alpha Vantage..."):
                stock_success = False
                news_success = False
                
                if fetch_stock_data:
                    stock_df, stock_error = fetch_alpha_vantage_stock_data(stock_ticker, alpha_vantage_api_key)
                    if stock_df is not None:
                        stock_success = True
                        st.sidebar.success(f"Successfully fetched stock data for {stock_ticker}")
                        
                        # Store in database
                        store_success, store_error = store_real_data_in_db(stock_df=stock_df)
                        if not store_success:
                            st.sidebar.error(f"Failed to store stock data: {store_error}")
                    else:
                        st.sidebar.error(stock_error)
                
                if fetch_news_data:
                    news_df, news_error = fetch_alpha_vantage_news(alpha_vantage_api_key)
                    if news_df is not None:
                        news_success = True
                        st.sidebar.success(f"Successfully fetched {len(news_df)} news items")
                        
                        # Store in database
                        store_success, store_error = store_real_data_in_db(news_df=news_df)
                        if not store_success:
                            st.sidebar.error(f"Failed to store news data: {store_error}")
                    else:
                        st.sidebar.error(news_error)
                
                if stock_success or news_success:
                    st.sidebar.info("Data successfully stored in database. You can now query it with the LLM agent.")

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

# --- GET FILTERED DATA ---
filtered_df = get_data_from_db(ticker, model)

# --- DISPLAY DATA ---
data_tab, news_tab, metrics_tab = st.tabs(["📊 Trade Data", "📰 Market News", "📈 Metrics"])

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
    
    # Show real stock data if available
    conn = sqlite3.connect(DB_PATH)
    real_stock_df = pd.read_sql("SELECT * FROM real_stock_data ORDER BY timestamp DESC LIMIT 30", conn)
    conn.close()
    
    if not real_stock_df.empty:
        st.subheader("Real Stock Data (Last 30 Days)")
        
        # Group by ticker
        tickers = real_stock_df['ticker'].unique()
        
        for ticker in tickers:
            ticker_data = real_stock_df[real_stock_df['ticker'] == ticker].sort_values('timestamp')
            
            # Calculate daily returns
            ticker_data['daily_return'] = ticker_data['close'].pct_change() * 100
            
            st.subheader(f"{ticker} Stock Performance")
            
            # Create two columns for price and volume
            price_col, return_col = st.columns(2)
            
            with price_col:
                st.line_chart(ticker_data.set_index('timestamp')['close'], use_container_width=True)
            
            with return_col:
                st.bar_chart(ticker_data.set_index('timestamp')['daily_return'], use_container_width=True)
    else:
        st.info("No real stock data available. Use the 'Fetch Real Data' option in the sidebar to get live market data.")

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

# Add real news-related prompts if we have news data
conn = sqlite3.connect(DB_PATH)
news_count = conn.execute("SELECT COUNT(*) FROM market_news").fetchone()[0]
conn.close()

if news_count > 0:
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