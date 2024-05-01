import streamlit as st
import numpy as np
import pandas as pd
import pandas_datareader as data
import matplotlib.pyplot as plt
from keras.models import load_model
import yfinance as yf

start= "2010-01-01"
end= "2024-04-30"

st.title("Stock Trend Prediction")

user_input= st.text_input("Enter Stock Ticker", "HDB")
df= yf.download(user_input, start, end)

st.subheader("Company Details")
company= yf.Ticker(user_input)
dict= company.info
bigd= pd.DataFrame.from_dict(dict, orient='index')
bigd= bigd.transpose()
#bigd= bigd.reset_index()
#st.write(bigd)
bigdn= bigd[['longName', 'sector', 'previousClose', 'dayLow', 'dayHigh', 'dividendYield', 'forwardPE', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow' ,'priceToBook', 'mostRecentQuarter', 'totalCash', 'totalCashPerShare', 'returnOnAssets', 'returnOnEquity' ]]
bigdn= bigd.transpose()

st.write(bigdn)


st.subheader('Data From 2010-2024')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig= plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.xlabel('Time(Year)')
plt.ylabel('Price(in USD)')
st.pyplot(fig)


data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


model= load_model('stock_model.h5')

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days,data_testing])
input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range(100 , input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i , 0])


x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)


scaler= scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


st.subheader('Predicted Price')
fig2= plt.figure(figsize=(12,6))
#plt.plot(y_test , 'b' ,label = 'Original Price')
plt.plot(y_predicted , 'g' , label = 'Predicted Price')
plt.xlabel('Time(in days)')
plt.ylabel('Price(in USD)')
plt.legend()
st.pyplot(fig2)

buy= bigd.iat[0, 100]
st.subheader(buy)