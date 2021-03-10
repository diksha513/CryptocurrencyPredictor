import numpy as np
import pandas as pd
import datetime as date
import requests
from pandas_datareader import data
from sklearn.linear_model import LinearRegression
import pickle

pd.options.display.float_format = '{:,.2f}'.format

# by using ALPHA VANTAGE API
# https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=CNY&apikey=demo&datatype=csv

class Crypto_prediction:
  def __init__(self,symbol):
    API_KEY = 'VQTPJ1YBJF3UBLZB'
    base_url = 'https://www.alphavantage.co/query?'
    params = {'function': 'DIGITAL_CURRENCY_DAILY',
              'symbol': symbol,
              'market': 'INR',
              'apikey': API_KEY
              }
    response_dict = requests.get(base_url, params=params).json()
    key_ar = list(response_dict.keys())
    if len(key_ar) == 1:
        self.mydf = "Time limit exceeded, Alpha Vantage's standard API call frequency is 5 calls per minute and 500 calls per day"
        return

    df = pd.DataFrame.from_dict(response_dict[key_ar[-1]], orient='index')

    mydf = df.drop(labels=['1b. open (USD)','2b. high (USD)','3b. low (USD)','4b. close (USD)','6. market cap (USD)'],axis=1)
    mydf = mydf[::-1]
    for each in mydf.columns:
      mydf[each]= mydf[each].astype(float)
    self.mydf = mydf

  def get_prediction(self):
    mydf = self.mydf
    if type(mydf) == str:
      pickle.dump(mydf, open('model.pkl','wb'))
      return mydf

    df_feature = mydf.drop(labels=['4a. close (INR)'],axis=1)[:-1]
    df_closing = mydf.drop(labels=['1a. open (INR)','2a. high (INR)','3a. low (INR)','5. volume'],axis=1)[1:]
    df_target_label = mydf.drop(labels=['4a. close (INR)'],axis=1)[-1:]

    df_feature_array = df_feature.to_numpy()
    df_closing_array = df_closing.to_numpy()
    df_target_label_array = df_target_label.to_numpy()

    Linearreg = LinearRegression()
    Linearreg.fit(df_feature_array, df_closing_array)

    dict_ = {
      'regressor': Linearreg,
      'target_label':df_target_label_array,
      'last_closing_pt':df_closing_array[-1]
    }
    pickle.dump(dict_, open('model.pkl','wb'))

    # Linearreg_prediction = Linearreg.predict(df_target_label_array)
    # return (df_closing_array[-1],Linearreg_prediction[0])


# """
# Bitcoin.
# Ethereum.
# Ripple XRP.
# Tron.
# Tether.
# """
# arr = ['BTC','ETC','XRP','TRX','USDT']

# for i in arr:
#   obj = Crypto_prediction(i)
#   print(obj.get_prediction())