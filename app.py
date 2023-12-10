import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yfin
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler as m
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


#-------------------------------------------------------------------------------------------------------------------------------------------

import streamlit as st

st.title('Stock Trend Prediction')
start = st.text_input('Enter START date', '2013-12-01')
end = st.text_input('Enter END date', '2023-12-01')
user_input =st.text_input('Enter Stock Ticker', 'TSLA')

df = yfin.download(user_input, start=start, end=end)
st.subheader(f"Data from {start[0:4]}-{end[0:4]}")
st.write(df.describe())

st.subheader( 'Closing Price vs Time chart')
fig= plt.figure(figsize =(12,6))
plt.plot (df['Close'],'b',label="Closing Price")
plt.xlabel('Time')
plt.ylabel('Price')
#plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100 Moving Average(MA)')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize =(12, 6))
plt.plot(ma100,'r',label="100 MA")
plt.plot(df['Close'],'b',label="Closing Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize =(12, 6))
plt.plot(ma100,'r',label="100 MA")
plt.plot(ma200,'g',label="200 MA")
plt.plot(df['Close'],'b',label="Closing Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array= scaler.fit_transform(data_training)



from joblib import load
model = load('rf_model.joblib')



past_100_days = data_training.tail(100)

final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i,0])
    y_test.append(input_data[i,0])
x_test,y_test=np.array(x_test),np.array(y_test)

y_predicted=model.predict(x_test)

scaler=scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test *scale_factor


st.subheader('Prediction vs Orignal')
fig2=plt.figure(figsize=(12,6))

plt.plot(y_test,'b',label ='Original Price' )
plt.plot(y_predicted,'r', label ='Predicted Price')

plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)