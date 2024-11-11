import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Set page config
st.set_page_config(page_title="Tesla Stock Analysis")

# Title
st.title("Tesla Stock Price Analysis")

try:
    # Load Data
    df = pd.read_csv('Tesla.csv')

    # Data Overview Section
    st.header("Data Overview")
    st.dataframe(df.head())
    st.write("Dataset Shape:", df.shape)

    # Basic Stock Information
    st.header("Basic Stock Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Average Close Price", f"${df['Close'].mean():.2f}")
    with col2:
        st.metric("Highest Price", f"${df['High'].max():.2f}")
    with col3:
        st.metric("Lowest Price", f"${df['Low'].min():.2f}")

    # Display recent price trends as a table
    st.header("Recent Price Trends")
    st.dataframe(df[['Date', 'Close']].tail(10))

    # Feature Engineering
    # Create technical indicators
    df['open-close'] = df['Open'] - df['Close']
    df['low-high'] = df['Low'] - df['High']
    df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    # Extract month from date for quarter end calculation
    df['month'] = pd.to_datetime(df['Date']).dt.month
    df['is_quarter_end'] = np.where(df['month']%3==0, 1, 0)

    # Model Training
    st.header("Price Movement Prediction Model")

    # Prepare features
    features = df[['open-close', 'low-high', 'is_quarter_end']].copy()
    target = df['target'].copy()

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, target, test_size=0.2, random_state=42
    )

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    # Display metrics
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Accuracy", f"{train_score:.3f}")
    with col2:
        st.metric("Test Accuracy", f"{test_score:.3f}")

    # Price Statistics
    st.header("Price Statistics")
    stats_df = pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
        'Value': [
            f"${df['Close'].mean():.2f}",
            f"${df['Close'].median():.2f}",
            f"${df['Close'].std():.2f}",
            f"${df['Close'].min():.2f}",
            f"${df['Close'].max():.2f}"
        ]
    })
    st.table(stats_df)

    # Prediction Interface
    st.header("Price Movement Prediction")
    col1, col2, col3 = st.columns(3)

    with col1:
        open_close = st.number_input("Open-Close Difference", value=0.0)
    with col2:
        low_high = st.number_input("Low-High Difference", value=0.0)
    with col3:
        is_quarter = st.selectbox("Is Quarter End?", [0, 1])

    if st.button("Predict"):
        input_data = scaler.transform([[open_close, low_high, is_quarter]])
        prediction = model.predict_proba(input_data)[0][1]
        st.write(f"Probability of price increase: {prediction:.2%}")

    # Monthly Analysis
    st.header("Monthly Analysis")
    monthly_avg = df.groupby('month')['Close'].mean().round(2)
    monthly_df = pd.DataFrame({
        'Month': range(1, 13),
        'Average Close Price': monthly_avg
    }).fillna(0)
    st.table(monthly_df)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please make sure the Tesla.csv file is in the correct location and contains the required columns.")

# Add footer
st.markdown("---")
st.markdown("Tesla Stock Price Analysis App")