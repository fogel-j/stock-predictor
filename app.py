import streamlit as st
from datetime import date
import yfinance
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Predictor")

stocks = ("GOOG", "APPL", "MSFT", "GME")
selected_stock = st.selectbox("Select stock", stocks)

n_years = st.slider("Years of prediction", 1,4)
period = n_years * 365

def load_data(stock):
    data = yfinance.download(stock, START, TODAY)
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stock)

st.subheader("Stock Data")
st.write(data.tail())

# Plot stock data
figure1 = go.Figure()
figure1.add_trace(go.Scatter(x=data['Date'], y=data['Open'],name='stock_open'))
figure1.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
figure1.layout.update(title_text="Time Series Data")
st.plotly_chart(figure1)

# Predict future prices
df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close":"y"})

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Plot forecast
st.subheader("Forecast Data")
figure2 = plot_plotly(m, forecast)
st.plotly_chart(figure2)