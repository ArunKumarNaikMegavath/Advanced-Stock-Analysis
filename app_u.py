import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import requests
from textblob import TextBlob
import time

# Set page config
st.set_page_config(page_title="Advanced Stock Analysis", layout="wide")

# Title and description
st.title("📈 Advanced Stock Analysis Dashboard")
st.markdown("Comprehensive stock analysis with technical indicators, market comparison, and sentiment analysis.")

# Sidebar for user inputs
st.sidebar.header("📊 Configuration")

# Stock symbol input
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., TSLA, AAPL):", "TSLA").upper()

# Date range selection
st.sidebar.subheader("Select Date Range")
end_date = datetime.now()
start_date = st.sidebar.date_input(
    "Start Date",
    end_date - timedelta(days=365)
)
end_date = st.sidebar.date_input("End Date", end_date)

# Technical Indicators Selection
st.sidebar.subheader("Technical Indicators")
show_sma = st.sidebar.checkbox("Show SMA", True)
show_rsi = st.sidebar.checkbox("Show RSI", True)
show_macd = st.sidebar.checkbox("Show MACD", True)
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", True)

# Function to calculate technical indicators
def calculate_technical_indicators(df):
    # SMA
    df['SMA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'])
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Lower'] = bollinger.bollinger_lband()
    df['BB_Middle'] = bollinger.bollinger_mavg()
    
    return df

# Function to fetch news and analyze sentiment
@st.cache_data(ttl=3600)
def get_news_sentiment(symbol):
    try:
        # Using Alpha Vantage News API (you'll need to get a free API key)
        api_key = "TZXOWD6JI8Z3K8HR"  # Replace with your API key
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}"
        response = requests.get(url)
        news_data = response.json()
        
        if 'feed' in news_data:
            sentiments = []
            for article in news_data['feed'][:10]:  # Analyze last 10 news articles
                blob = TextBlob(article['title'] + " " + article.get('summary', ''))
                sentiments.append(blob.sentiment.polarity)
            
            return {
                'average_sentiment': np.mean(sentiments),
                'num_articles': len(sentiments),
                'positive_news': sum(1 for s in sentiments if s > 0),
                'negative_news': sum(1 for s in sentiments if s < 0)
            }
    except Exception as e:
        st.warning(f"Could not fetch news data: {str(e)}")
        return None

# Function to create interactive price chart
def create_price_chart(df):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05,
                        row_heights=[0.6, 0.2, 0.2])
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ), row=1, col=1)
    
    # Add technical indicators
    if show_sma:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50', line=dict(color='blue')), row=1, col=1)
    
    if show_bollinger:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)
    
    # RSI
    if show_rsi:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        height=800,
        title=f'{stock_symbol} Stock Analysis',
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2_title="Volume",
        yaxis3_title="RSI"
    )
    
    return fig

# Retry logic for fetching market data
def fetch_market_data():
    retries = 3  # Number of retry attempts
    for attempt in range(retries):
        try:
            market_df = yf.download('^GSPC', start=start_date, end=end_date)
            if not market_df.empty:
                return market_df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)  # Wait 2 seconds before retrying
            else:
                st.warning("Market data could not be retrieved after multiple attempts. Continuing without S&P 500 data.")
                return None

# Main app
try:
    # Load data
    with st.spinner(f'Fetching data for {stock_symbol}...'):
        stock = yf.Ticker(stock_symbol)
        df = stock.history(start=start_date, end=end_date)
        market_df = fetch_market_data()  # Using the retry mechanism

        # Check if stock data is fetched correctly
        if df.empty:
            st.error(f"Could not fetch data for {stock_symbol}. Please check the symbol and try again.")
        else:
            # Calculate technical indicators
            df = calculate_technical_indicators(df)
            
            # Company Info
            company_info = stock.info
            
            # Display company information
            st.header("Company Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Company", company_info.get('longName', stock_symbol))
            with col2:
                st.metric("Sector", company_info.get('sector', 'N/A'))
            with col3:
                st.metric("Market Cap", f"${company_info.get('marketCap', 0):,.0f}")
            
            # Price Chart
            st.header("Technical Analysis")
            fig = create_price_chart(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Market Comparison
            st.header("Market Comparison")
            if market_df is not None and not df.empty:
                # Align the indices of df and market_df
                df, market_df = df.align(market_df, join='inner', axis=0)
                
                # Check if the dataframes have any rows to compare
                if not df.empty and not market_df.empty:
                    # Calculate normalized returns for comparison
                    comparison_df = pd.DataFrame({
                        'Stock': (df['Close'] / df['Close'].iloc[0]) * 100,
                        'S&P 500': (market_df['Close'] / market_df['Close'].iloc[0]) * 100
                    })
                    st.line_chart(comparison_df)
                else:
                    st.warning("The stock or S&P 500 data has insufficient data points for comparison. Please adjust the date range.")
            else:
                st.warning("Market data is unavailable. S&P 500 comparison will not be displayed.")
            
            # Technical Analysis Summary
            st.header("Technical Analysis Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("SMA & Current Price")
                if len(df) > 0:
                    sma20 = df['SMA20'].iloc[-1] if not pd.isna(df['SMA20'].iloc[-1]) else None
                    sma50 = df['SMA50'].iloc[-1] if not pd.isna(df['SMA50'].iloc[-1]) else None
                    current_price = df['Close'].iloc[-1] if not pd.isna(df['Close'].iloc[-1]) else None
                    
                    if sma20 and sma50 and current_price:
                        st.write(f"SMA 20: {sma20:.2f}")
                        st.write(f"SMA 50: {sma50:.2f}")
                        st.write(f"Current Price: {current_price:.2f}")
                        
                        if sma20 > sma50:
                            st.success("Bullish crossover!")
                        else:
                            st.error("Bearish crossover!")
                    else:
                        st.warning("Unable to calculate SMA or current price.")
            
            # Sentiment Analysis
            st.header("Sentiment Analysis")
            sentiment_data = get_news_sentiment(stock_symbol)
            if sentiment_data:
                st.write(f"Average Sentiment: {sentiment_data['average_sentiment']:.2f}")
                st.write(f"Positive News: {sentiment_data['positive_news']} articles")
                st.write(f"Negative News: {sentiment_data['negative_news']} articles")
            else:
                st.warning("No sentiment data available.")
        
except Exception as e:
    st.error(f"An error occurred: {e}")
