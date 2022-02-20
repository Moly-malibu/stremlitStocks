#base
import streamlit as st
import streamlit.components.v1 as components
import sqlite3
import pandas as pd

#Index
from numpy.lib.shape_base import column_stack
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import datetime
import requests
from requests import get
from bs4 import BeautifulSoup

#Profit
# from pandas_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report
# from pandas_profiling.utils.cache import cache_file

#Portfolio
import numpy as np
import yfinance 
import yfinance as yf 
yf.pdr_override()
plt.style.use('fivethirtyeight')

#Prediction
import math
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
# tf.keras.models.Model()
# from tensorflow.keras.layers import layers
import keras
from keras.layers import Dense, LSTM
from bs4 import BeautifulSoup
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.compat.v1 
# tf.compat.v1.get_default_graph()

#statement
from re import sub
from pandas.core.reshape.merge import merge
from yahoofinancials import YahooFinancials
from yahoo_fin.stock_info import get_data
import yahoo_fin.options as ops
import yahoo_fin.stock_info as si
from matplotlib import style
style.use('ggplot')

#Stocks
from numpy.lib.shape_base import split
from pandas._config.config import reset_option
from pandas.core import groupby
from requests.api import options 
import os


def main():
    # Register pages
    pages = {
        "Home": Home,
        "Index": Index,
        "Stock": Stock,
        'Statement': Statement,
        'Portfolio': Portfolio,
        "Prediction_model": Prediction_model,
        # "Profit": Profit,
    }
    st.sidebar.title("Companies Analysis")
    page = st.sidebar.selectbox("Select Menu", tuple(pages.keys()))
    pages[page]()

def Home():
    def main():
        st.markdown("<h1 style='text-align: center; color: #002966;'>Finances and Stocks</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: #002966;'>App for Streamlines Decisions</h1>", unsafe_allow_html=True)
        st.write(
        """
        Artificial Intelligence helps you to perform a successful future, taking the correct decisions! 
        
        Here you will get Financial information of more than *** 8,100 companies***, Fast, Easy, and Simple  
        
        Making decisions on constructing the portfolio used the Algorithms that brings potential accurancy to trading:
        - ***Chart Analysis of single and multiple companies' stocks***.  
        - ***Machine Learn Forecasting:***  
                - Compared Forecasting
                - Long Short Term Memory.
                - Decision Tree Regression.
                - Linear Regression.
        - ***Portfolio:*** 
                - Stock Return, 
                - Correlation, 
                - Covariance Matrix for return.
                - Variance.
                - Volatility, 
                - Dayily Expected Porfolio Return.
                - Annualised Portfolio Return.
                - Growth of investment. 
        - ***Financial Analysis:***
                
                - Ratios, 
                - Monte Carlo Simulation
                - Cash Flow
                - Income Statement
                - Balance Sheet
                - Quote Table.
                - Call Option.
                
        - ***Financial Information:***
                - Company Information.
                - Company Share Asigned.
                - Stocks Recommendations.
                - Actions and Split.
                - Statistics.
                - Status of Evaluation.
        - ***Profiling each company:***
                - Interactions in High, Low, Close, Volume and Dividens.
                - Correlations: Pearson's r, Spearman's p, Kendalls's T, Phik (@K)
                - Matrix.
                - Heatmap.
                - Dentrogram.
        ---
        """)
        today = st.date_input('Today is', datetime.datetime.now())
        page_bg_img = '''
            <style>
            body {
            background-image: url("https://images.pexels.com/photos/1024613/pexels-photo-1024613.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=1000");
            background-size: cover;
            }
            </style>
            '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
        footer_temp1 = """
            <!-- CSS  -->
            <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css" type="text/css" rel="stylesheet" media="screen,projection"/>
            <link href="static/css/style.css" type="text/css" rel="stylesheet" media="screen,projection"/>
            <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.5.0/css/all.css" integrity="sha384-B4dIYHKNBt8Bc12p+WXckhzcICo0wtJAoU8YZTY5qE0Id1GSseTk6S+L3BlXeVIU" crossorigin="anonymous">
            <footer class="background-color: red">
                <div class="container" id="About App">
                <div class="row">
                    <div class="col l6 s12">
                    </div>
            <div class="col l3 s12">
                    <h5 class="black-text">Connect With Me</h5>
                    <ul>
                        <a href="http://www.monicadatascience.com/" target="#002966" class="black-text">
                        ❤<i class="❤"></i>
                    </a>
                    <a href="https://www.linkedin.com/in/monica-bustamante-b2ba3781/3" target="#002966" class="black-text">
                        <i class="fab fa-linkedin fa-4x"></i>
                    <a href="http://www.monicadatascience.com/" target="#002966" class="black-text">
                    ❤<i class="❤"></i>
                    </a>
                    <a href="https://github.com/Moly-malibu/financesApp" target="#002966" class="black-text">
                        <i class="fab fa-github-square fa-4x"></i>
                    <a href="http://www.monicadatascience.com/" target="#002966" class="black-text">
                    ❤<i class="❤"></i>
                    </a>
                    </ul>
                    </div>
                </div>
                </div>
                <div class="footer-copyright">
                <div class="container">
                Made by <a class="black-text text-lighten-3" href="http://www.monicadatascience.com/">Monica Bustamante</a><br/>
                <a class="black-text text-lighten-3" href=""> @Copyrigh</a>
                </div>
                </div>
            </footer>
            """
        components.html(footer_temp1,height=500)

    if __name__ == "__main__":
        main()

title_temp = """
	 <!-- CSS  -->
	  <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
	  <link href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css" type="text/css" rel="stylesheet" media="screen,projection"/>
	  <link href="static/css/style.css" type="text/css" rel="stylesheet" media="screen,projection"/>
	   <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.5.0/css/all.css" integrity="sha384-B4dIYHKNBt8Bc12p+WXckhzcICo0wtJAoU8YZTY5qE0Id1GSseTk6S+L3BlXeVIU" crossorigin="anonymous">
	 <footer class="background-color: red">
	    <div class="container" id="About App">
	      <div class="row">
	        <div class="col l6 s12">
                <h5 class="black-text">Artificial Intelligence</h5>
	          <p class="grey-text text-lighten-4">Using Streamlit, Yahoo Finances, Sklearn, Tensorflow,  Keras, Pandas Profile, Numpy, Math, Data Visualization. </p>
	        </div>     
	  </footer>
	"""
components.html(title_temp,height=100)
    
#Analysis stocks companies by close and volume.
def Index(): 
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://img.freepik.com/free-photo/3d-geometric-abstract-cuboid-wallpaper-background_1048-9891.jpg?size=626&ext=jpg&ga=GA1.2.635976572.1603931911");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    symbols = 'https://raw.githubusercontent.com/Moly-malibu/Stocks/main/bxo_lmmS1.csv'
    df = pd.read_csv(symbols)
    st.markdown("<h1 style='text-align: center; color: #002966;'>Stock Price </h1>", unsafe_allow_html=True)
    start = st.sidebar.date_input("Enter Date Begin Analysis: ") 
    tickerSymbol = st.sidebar.selectbox('Stocks Close and Volume price by Company', (df))
    tickerData = yf.Ticker(tickerSymbol)
    tickerDf = tickerData.history(period='id', start=start, end=None)
    st.write("""# Analysis of Data""")
    st.write("""
    ## Closing Price
    """)
    st.line_chart(tickerDf.Close)
    st.write(""" 
    ## Volume Price
    """)
    st.line_chart(tickerDf.Volume)
    st.markdown("<h1 style='text-align: center; color: #002966;'>Stock Price Compared</h1>", unsafe_allow_html=True)
    st.write("""
    **Business** and **Techology** are two fills that have changed the world, both occupy the main ratings in finance, being one of the most highly valued in the stock market leading their owners to be billionaires, in this simple application we can analyze the stock movement and prediction of future price of stock used algoriths and Machile Learning.
    Show are the Stock **Closing Price** and ** Volume** of Stocks by year!
    """)
    st.markdown('Help to take algoritmc decision about stocks')
    company = tickerSymbol1 = st.sidebar.multiselect("Select Companies Stock be compared", (df))
    if company:
        st.subheader("""**Compared Status**""")
        button_clicked = st.sidebar.button("GO")
        analysis = yf.download(tickerSymbol1, start=start, end=None)
        st.write('Analysis', analysis)
        analysis['Adj Close'].plot()
        plt.xlabel("Date")
        plt.ylabel("Adjusted")
        plt.title("Company Stock")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

# #Portfolio
def Portfolio():
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://images.pexels.com/photos/2748756/pexels-photo-2748756.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=1000");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    symbols = 'https://raw.githubusercontent.com/Moly-malibu/Stocks/main/bxo_lmmS1.csv'
    df = pd.read_csv(symbols)
    st.markdown("<h1 style='text-align: center; color: #002966;'>Portfolio</h1>", unsafe_allow_html=True)
    st.write(""" Make your ***own Portfolio*** with 5 companies and analyze what will be your profit.""")
    st.write("""***Instructions:***""") 
    st.write(
        """
        - Select 5 companies where you want to invest or Analysis.  ('others' it needs more companies)  
        - Select Date. (NAN empty date)
        ---
        """)
    
    stockStarData = st.sidebar.date_input("Select Date when you started to investing:")
    company = tickerSymbol1 = st.sidebar.multiselect("Select only 5 Companies to create the Portfolio", (df['Symbol']))
    button_clicked = st.sidebar.button("GO")
    if company:
        def getmyportfolio(stock=tickerSymbol1, start=stockStarData, end=None):
            numAssets = len(tickerSymbol1)
            st.write('***you have*** ' +str(numAssets) + ' ***Assets in your Portafolio.***')
            data = yf.download(tickerSymbol1, start=start, end=end)['Adj Close']
            return data
        my_stocks = getmyportfolio(tickerSymbol1)
        st.write(my_stocks)
        daily_return = my_stocks.pct_change(1)
        daily_return.corr()
        daily_return.cov()
        daily_return.var()
        daily_return.std()
        st.write('***Stock Return ***',daily_return)
        st.write('***Stock Correlation ***',daily_return.corr())
        st.write('***Stock Covariance Matrix for Return***',daily_return.cov())
        st.write('***Stock Variance ***',daily_return.var())
        st.write('***Stock Volatility ***', daily_return.std())
    #Visualization
        plt.figure(figsize=(12, 4.5))
        for c in daily_return.columns.values:
            plt.plot(daily_return.index, daily_return[c], lw=2, label=c)
        plt.legend(loc='upper right', fontsize=10)
        plt.title('Volatility')
        plt.ylabel('Dayly Return')
        plt.xlabel('Date')
        plt.style.use('dark_background')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    #get Growth Investment
        dailyMeanSimpleReturns = daily_return.mean()
        st.write('***Daily Mean Simple Return:*** ', dailyMeanSimpleReturns)
        randomWeights = np.array([0.4, 0.1, 0.3, 0.1, 0.1])
        portfoliosimpleReturn = np.sum(dailyMeanSimpleReturns*randomWeights)
        st.write('***Daily Expected Portfolio Return:*** '+str(portfoliosimpleReturn))
        st.write('***Expected Annualised Portfolio Return:*** ' + str(portfoliosimpleReturn*253))
        dailyCumulSimpleReturn = (daily_return+1).cumprod()
        st.write('***Growth of Investment:*** ', dailyCumulSimpleReturn)
    #Visualization
        plt.figure(figsize=(12.2, 4.5))
        for c in dailyCumulSimpleReturn.columns.values:
            plt.plot(dailyCumulSimpleReturn.index, dailyCumulSimpleReturn[c], lw=2, label=c)
        plt.legend(loc='upper left', fontsize=10)
        plt.xlabel('Date')
        plt.ylabel('Growth fo $1 Investment')
        plt.title('Daily Cumulative Returns')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

# #Differente models to predict the price.

def Prediction_model():
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://images.pexels.com/photos/4194857/pexels-photo-4194857.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=1000");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    symbols = 'https://raw.githubusercontent.com/Moly-malibu/Stocks/main/bxo_lmmS1.csv'
    df = pd.read_csv(symbols)
    # #Firs model to predict price and accuracy
    now = pd.datetime.now()
    tickerSymbol = st.sidebar.selectbox('Company List', (df['Symbol']))
    tickerData = yf.Ticker(tickerSymbol)
    tickerDf = tickerData.history(period='id', start='2019-01-01', end=now)
    data = tickerDf.filter(['Close'])
    dataset = data.values

    company_hist = st.sidebar.checkbox('Long Short Term Memory')
    if company_hist:
        st.markdown("<h1 style='text-align: center; color: #002966;'>Long Short Term Memory</h1>", unsafe_allow_html=True)
        #Scaler data
        train_len = math.ceil(len(dataset)*.8)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)
        train_data = scaled_data[0:train_len, :]
        #train data
        x_train = []
        y_train = []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i,0])
            y_train.append(train_data[i,0])
            if i<=60:
                print(x_train)
                print(y_train)
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))        #Model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=1, epochs=1)
        #Test data
        test_data = scaled_data[train_len - 60: , :]
        x_test = []
        y_test = dataset[train_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i,0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        #Graphic
        train = data[:train_len]
        valid = data[train_len:]
        valid['Predictions'] = predictions
        plt.figure(figsize=(16,8))
        plt.title('Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD($)', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['train', 'Val', 'Predictions'], loc='upper left')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        st.markdown("<h1 style='text-align: center; color: #002966;'>Forecasting the Price Stocks</h1>", unsafe_allow_html=True)
        st.write(""" 
        Using keras Long Short Term Memory (LSTM) model that permit to store past information to predict the future price of stocks.
        """)
        st.write(predictions)
        st.markdown("<h1 style='text-align: center; color: #002966;'>Root Mean Square Deviation</h1>", unsafe_allow_html=True)
        st.write(""" 
        The RMSE shows us how concentrated the data is around the line of best fit.
        """)
        rmse = np.sqrt(np.mean(predictions - y_test)**2)
        st.write(rmse)
    #Second Model
    company_hist = st.sidebar.checkbox('Decision Tree Regression')
    if company_hist: 
        forcast_days = 25
        tickerDf['Prediction'] = tickerDf[['Close']].shift(-forcast_days)
        X=np.array(tickerDf.drop(['Prediction'], 1)[:-forcast_days].fillna(0))
        y=np.array(tickerDf['Prediction'])[:-forcast_days] 
        # #Train Data    
        x_train, x_test, y_train, y_test= train_test_split(X,y, test_size=0.25)
        tree = DecisionTreeRegressor().fit(x_train, y_train)
        x_future = tickerDf.drop(['Prediction'], 1)[:-forcast_days]
        x_future = x_future.tail(forcast_days)
        x_future = np.array(x_future)
        tree_prediction = tree.predict(x_future)
        st.markdown("<h1 style='text-align: center; color: #002966;'>Decision Tree Regression Model</h1>", unsafe_allow_html=True)
        # #Graph 
        predictions = tree_prediction
        valid = tickerDf[X.shape[0]:]
        valid['Predictions'] = predictions
        plt.figure(figsize=(16,8))
        plt.title('Model')
        plt.xlabel('Days')
        plt.ylabel('Close Price USD($)')
        plt.plot(tickerDf['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['orig', 'Val', 'Pred'])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        st.write('Prediction:', predictions) 
        st.write('Accuracy:', tree.score(x_train, y_train))
        tree_confidence = tree.score(x_test, y_test)
        st.write('Confidence:', tree_confidence)
    # Third Model
    company_hist = st.sidebar.checkbox('Linear Regression')
    if company_hist: 
        st.markdown("<h1 style='text-align: center; color: #002966;'>Linea Regression Model</h1>", unsafe_allow_html=True)
        forcast_days = 25
        tickerDf['Prediction'] = tickerDf[['Close']].shift(-forcast_days)
        X=np.array(tickerDf.drop(['Prediction'], 1)[:-forcast_days].fillna(0))
        y=np.array(tickerDf['Prediction'])[:-forcast_days] 
        x_future = tickerDf.drop(['Prediction'], 1)[:-forcast_days]
        x_future = x_future.tail(forcast_days)
        x_future = np.array(x_future)
        # #Train Data    
        x_train, x_test, y_train, y_test= train_test_split(X,y, test_size=0.25)
        lr = LinearRegression().fit(x_train, y_train)
        lr_prediction = lr.predict((x_future))
        lr_confidence = lr.score(x_test, y_test)
        #Prediction
        predictions = lr_prediction
        valid = tickerDf[X.shape[0]:]
        valid['Predictions'] = predictions
        plt.figure(figsize=(16,8))
        plt.title('Model')
        plt.xlabel('Days')
        plt.ylabel('Close Price USD($)')
        plt.plot(tickerDf['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['orig', 'Val', 'Pred'])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        st.write('Predictioin by LR:', predictions)
        st.write('Accuracy:', lr.score(x_train, y_train))
        st.write('linear Regression confidence:', lr_confidence)
    st.markdown("<h1 style='text-align: center; color: #002966;'>Compared Forecasting</h1>", unsafe_allow_html=True)
    new_predict = tickerDf['Close']
    st.write(tickerDf)
    # ...

# def Profit():
#     page_bg_img = '''
#     <style>
#     body {
#     background-image: url("https://img.freepik.com/free-photo/3d-geometric-abstract-cuboid-wallpaper-background_1048-9891.jpg?size=626&ext=jpg&ga=GA1.2.635976572.1603931911");
#     background-size: cover;
#     }
#     </style>
#     '''
#     st.markdown(page_bg_img, unsafe_allow_html=True)
#     st.markdown("<h1 style='text-align: center; color: #002966;'>Profiling Report</h1>", unsafe_allow_html=True)
#     symbols = 'https://raw.githubusercontent.com/Moly-malibu/Stocks/main/bxo_lmmS1.csv'
#     df = pd.read_csv(symbols)
#     tickerSymbol = st.sidebar.selectbox('Company List', (df))
#     company = yf.Ticker(tickerSymbol)
#     analysis = company.history(period='max', interval='1wk')
#     profile = ProfileReport(analysis, explorative=True)
#     st_profile_report(profile)

def Statement():
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://images.pexels.com/photos/2748757/pexels-photo-2748757.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=1000");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    symbols = 'https://raw.githubusercontent.com/Moly-malibu/Stocks/main/bxo_lmmS1.csv'
    df = pd.read_csv(symbols)
    ticker = st.sidebar.selectbox('Stocks by Company', (df))
    tickerData = YahooFinancials(ticker)
    company = yf.Ticker(ticker)
    # st.write(company.info)
    company_general = st.sidebar.checkbox("Financial Ratio")
    if company_general:
        st.markdown("<h1 style='text-align: center; color: #002966;'>Financial Ratios</h1>", unsafe_allow_html=True)
        st.write('***Payout Ratio:*** ', company.info["payoutRatio"])
        st.write('***Trailing Annual Dividend Yield:*** ', company.info["trailingAnnualDividendYield"])
        st.write('***Dividend Rate:*** ', company.info["dividendRate"])
        st.write('***Profit Margins: ***', company.info["profitMargins"])
        st.write('***Peg Ratio: ***', company.info["pegRatio"])
        yahoo_financials = YahooFinancials(ticker)
        marketcap = yahoo_financials.get_market_cap()
        price_to_sales = yahoo_financials.get_current_price()
        dividend_yield = yahoo_financials.get_dividend_yield()
        income_balance=si.get_income_statement(ticker)
        transpose_income=income_balance.transpose()
        balance_income=si.get_balance_sheet(ticker)
        transpose_balance=balance_income.transpose()
        st.write("""**Dividends**""", company.dividends) 
        income=si.get_income_statement(ticker)
        transpose=income.transpose()
        interest_coverage1 = transpose['operatingIncome'] 
        interest_coverage2 = transpose['interestExpense']
        st.write('***Interest Coverage:*** Operating Income / interest Expenses', interest_coverage1/interest_coverage2)
        gross_profit_margin1 = transpose['totalRevenue'] 
        gross_profit_margin2 = transpose['costOfRevenue']
        st.write('***Gross Profit Margin:*** Total Revenue / Gross Profit Margin',(gross_profit_margin1-gross_profit_margin2)/gross_profit_margin1)
        balance=si.get_balance_sheet(ticker)
        transpose=balance.transpose()
        current_ratio1 = transpose['totalCurrentAssets'] 
        current_ratio2 = transpose['totalCurrentLiabilities']
        debt_to_assets1 = transpose['otherCurrentAssets'] 
        debt_to_assets2 = transpose['totalAssets']
        st.write('***Debit Assets:*** Total Debit / Total Assets', (debt_to_assets1/debt_to_assets2))
        debt_to_equity1 = transpose['otherCurrentAssets'] 
        debt_to_equity2 = transpose['totalStockholderEquity']
        st.write('***Debit to Equity:*** Total Debit / Total Stock Holders Equity', (debt_to_equity1/debt_to_equity2))
        ROE1 = transpose_income['netIncome'] 
        ROE2 = transpose_balance['totalStockholderEquity']
        st.write('***Return On Equity ROE:*** Net Income / (Total Stock Holder Equity + Total Stock Holder Equity)/2',(ROE1/((ROE2+ROE2)/2)))
        ROA1 = transpose_income['netIncome'] 
        ROA2 = transpose_balance['totalAssets']
        st.write('***Return On Assets:*** Net Income / Total Assets',(ROA1/ROA2))

    company_simulation = st.sidebar.checkbox("Monte Carlo Simulation")
    if company_simulation:
        st.markdown("<h1 style='text-align: center; color: #002966;'>Monte Carlo Simulation Price</h1>", unsafe_allow_html=True)
        st.write("""Monte Carlo Simulation project future price for the stocks. """)
        yahoo_financials = YahooFinancials(ticker)
        price = yahoo_financials.get_current_price()
        st.write('***Current Price:***', price)
        marketcap = yahoo_financials.get_market_cap()
        st.write('***Market Capital***', marketcap)
        income_balance=si.get_income_statement(ticker)
        transpose_income=income_balance.transpose()
        revenue = transpose_income['totalRevenue'] 
        st.write('***Price to sales:*** (Market Capital / Revenue', marketcap/revenue)
        price_to_earnings = transpose_income['netIncome'] 
        st.write('***Price to Earnings:*** (Market Capital/ Net Income', marketcap/price_to_earnings)
        balance_income=si.get_balance_sheet(ticker)
        transpose_balance=balance_income.transpose()
        price_to_book = transpose_balance['totalStockholderEquity']
        st.write('***Price to book:*** (marketcap/Total Stock Holders Equity', marketcap/price_to_book)
        start = st.date_input("Please enter date begin Analysis: ")
        price = yf.download(ticker, start=start, end=None)['Close']
        returns = price.pct_change()
        last_price = price[-1]
        num_simulations = 1000
        num_days = 252
        num_simulations_df = pd.DataFrame()
        for x in range(num_simulations):
            count = 0
            daily_vol = returns.std()
            price_series = []
            price = last_price*(1+np.random.normal(0,daily_vol))
            price_series.append(price)
            for y in range(num_days):
                if count == 251:
                    break
                price = price_series[count] * (1+np.random.normal(0,daily_vol))
                price_series.append(price)
                count +=1
            num_simulations_df[x] = price_series
        fig = plt.figure()
        plt.title('Monte Carlo Simulation')
        plt.plot(num_simulations_df)
        plt.axhline(y=last_price, color='r', linestyle='-')
        plt.xlabel('Day')
        plt.ylabel('Price')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        st.write('Price Series Predict: ', num_simulations_df)
    # company_general = st.sidebar.checkbox("Quick_Ratio")
    # if company_general:
    #     st.subheader("""**Quick Ratio**""")
    #     balance=si.get_balance_sheet(ticker)
    #     transpose=balance.transpose()
    #     quick_ratio1 = transpose['otherCurrentAssets'] 
    #     quick_ratio2 = transpose['inventory'] 
    #     quick_ratio3 = transpose['otherCurrentLiab']
    #     quick_ratio = ((quick_ratio1-quick_ratio2)/quick_ratio3)
    #     if not quick_ratio2:
    #         st.write("No data available")
    #     else:
    #         st.write('(***Quick Ratio:*** CurrentAssets - Inventory)/Current Liabilities)', (quick_ratio1-quick_ratio2)/quick_ratio3)
    company_hist = st.sidebar.checkbox("Cash Flow")
    if company_hist:
            st.markdown("<h1 style='text-align: center; color: #002966;'>Cash Flow</h1>", unsafe_allow_html=True)
            display_cash = si.get_cash_flow(ticker)
            if display_cash.empty == True:
                st.write("No data available")
            else:
                st.write(display_cash)
    company_hist = st.sidebar.checkbox("Income Statement")
    if company_hist:
            st.markdown("<h1 style='text-align: center; color: #002966;'>Income Statement</h1>", unsafe_allow_html=True)
            display_income_stat = si.get_income_statement(ticker)
            if display_income_stat.empty == True:
                st.write("No data available")
            else:
                st.write(display_income_stat)
    company_hist = st.sidebar.checkbox("Balance Sheet")
    if company_hist:
            st.markdown("<h1 style='text-align: center; color: #002966;'>Balance Sheet</h1>", unsafe_allow_html=True)
            display_balance = si.get_balance_sheet(ticker)
            if display_balance.empty == True:
                st.write("No data available")
            else:
                st.write(display_balance)
    company_hist = st.sidebar.checkbox("Quote Table")
    if company_hist:
            st.markdown("<h1 style='text-align: center; color: #002966;'>Quote Table</h1>", unsafe_allow_html=True)
            display_table = si.get_quote_table(ticker, dict_result=False)
            if display_table.empty == True:
                st.write("No data available")
            else:
                st.write(display_table)
            quote_table = si.get_quote_table(ticker)
            t = quote_table["Forward Dividend & Yield"]
            st.write('Forward Dividend & Yield:', t)
            display_capital = si.get_quote_table(ticker)["Market Cap"]
            st.write('Market Capital', display_capital)     
    company_hist = st.sidebar.checkbox("Call Option")
    if company_hist:
            st.markdown("<h1 style='text-align: center; color: #002966;'>Call Option</h1>", unsafe_allow_html=True)
            c= ops.get_calls(ticker)
            transpose = c.transpose() 
            st.write(transpose) 
        # ...
def Stock():
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://images.pexels.com/photos/1353938/pexels-photo-1353938.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=1000");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    symbols = 'https://raw.githubusercontent.com/Moly-malibu/Stocks/main/bxo_lmmS1.csv'
    df = pd.read_csv(symbols)
    st.markdown("<h1 style='text-align: center; color: #002966;'>Financial Information</h1>", unsafe_allow_html=True)
    st.write("""
    Financial information from the companies and Stocks by years!
    """)
    start = st.sidebar.date_input("Date to Analysis")
    st.sidebar.subheader("Index")
    tickerSymbol2 = st.sidebar.selectbox('Stocks by Company', (df))
    tickerData = yf.Ticker(tickerSymbol2)
    tickerDf = tickerData.history(period='id', start=start, end=None)
    company = yf.Ticker(tickerSymbol2)
    # st.write(company.info)
    company_general = st.sidebar.checkbox("Company Information")
    if company_general:
        st.markdown("<h1 style='text-align: center; color: #002966;'>General Information</h1>", unsafe_allow_html=True)
        st.write('***Sector:*** ', company.info["sector"])
        st.write('***Industry:*** ', company.info["industry"])
        st.write('***Phone:*** ', company.info["phone"])
        st.write('***Address: ***', company.info["address1"])
        st.write('***City: ***', company.info["city"])
        st.write('***Country: ***', company.info["country"])
        st.write('***Web: ***', company.info["website"])
        st.write('***Business Summary:*** ', '\n', company.info["longBusinessSummary"])
        st.write('***Job Generator***', company.info["fullTimeEmployees"])
    company_hist = st.sidebar.checkbox("Company Shares Asigned")
    if company_hist:
            st.markdown("<h1 style='text-align: center; color: #002966;'>Company Shares  Asigned </h1>", unsafe_allow_html=True)
            display_histo = company.major_holders
            display_mh = company.history(period='max')
            if display_histo.empty == True:
                st.write("No data available")
            else:
                st.write(display_histo)
    company_recomend = st.sidebar.checkbox("Stocks Recommendations")
    if company_recomend:
            st.markdown("<h1 style='text-align: center; color: #002966;'>Stocks Recommendatios</h1>", unsafe_allow_html=True)    
            display_recomend = company.recommendations
            if display_recomend.empty == True:
                st.write("No data available")
            else:
                st.write(display_recomend)
    company_job = st.sidebar.checkbox("Action and Split")
    if company_job:
        st.markdown("<h1 style='text-align: center; color: #002966;'>History Actions and Split</h1>", unsafe_allow_html=True)
        data = {}
        list = [(tickerSymbol2)]
        for ticker in list:
            ticker_object = yf.Ticker(ticker)
            temp = pd.DataFrame.from_dict(ticker_object.info, orient='index')
            temp.reset_index(inplace=True)
            temp.columns = ['Attribute', 'Recent']
            data[ticker] = temp
        merge = pd.concat(data)
        merge = merge.reset_index()
        del merge['level_1']
        merge.columns=['Ticker', 'Attribute', 'Recent'] 
        split=company.history(period='max', interval='1wk')
        st.sidebar.checkbox("Stock level")
        st.write('Company History', split)
    company_stadistic = st.sidebar.checkbox("Statistics")
    if company_stadistic:
        st.markdown("<h1 style='text-align: center; color: #002966;'>Statistics</h1>", unsafe_allow_html=True)
        data = yf.download((tickerSymbol2), start=start, end=None, group_by='tickers')
        table = st.table(data.describe())
    company_hist = st.sidebar.checkbox("Status of Evaluation")
    if company_hist:
            st.markdown("<h1 style='text-align: center; color: #002966;'>Status of Evaluation</h1>", unsafe_allow_html=True)
            display_evaluation = si.get_stats_valuation(tickerSymbol2)
            if display_evaluation.empty == True:
                st.write("No data available")
            else:
                st.write(display_evaluation)
if __name__ == "__main__":
   main()

