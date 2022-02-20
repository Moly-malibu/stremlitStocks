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
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling.utils.cache import cache_file

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
