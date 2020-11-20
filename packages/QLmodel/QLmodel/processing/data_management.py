import numpy as np
import pandas as pd
import quandl
import yfinance as yf
import datetime as dt
import os

from QLmodel.config import config

def create_data():
	symbols_table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
								 header=0)[0]
	symbols = list(symbols_table.loc[:, "Symbol"])
	symbols.append('SPY')

	for symbol in symbols:
		if not os.path.exists(f"data/{symbol}.csv"):
			try:
				data = yf.download(symbol, start=config.START_DATE, end=config.END_DATE)
				if data.size > 0:
					data.to_csv(f'{config.DATASET_DIR}/{symbol}.csv')
				else:
					print("Not saving...")
			except:
				print("Stock not found. Not saving...")



def load_data(symbols=None):

	dates = pd.date_range(start = config.START_DATE, end = config.END_DATE, freq='1D')
	df = pd.DataFrame()
	df['Date'] = dates
	df = df.set_index('Date')

	if not symbols:
		symbols = []
		for fileName in files:
			fileName = file.split('\\')[-1]
			fileName = file.split('.')[0]
			symbols.append(fileName)

	for symbol in symbols:

		temp = pd.read_csv(f'{config.DATASET_DIR}/{symbol}.csv')
		temp = temp[['Date','Adj Close']]
		temp.columns = ['Date', str(symbol)]
		temp['Date'] = pd.to_datetime(temp['Date'], format='%Y-%m-%d')
		temp = temp.set_index('Date')
		df = df.join(temp, how='left' )

	df.dropna(axis=0, how='all', inplace=True)
	df.fillna(value=0, axis=1, inplace=True)

	df_returns = pd.DataFrame()
	for name in df.columns:
		df_returns[name] = np.log(df[name]).diff()

	# split into train and test
	Ntest = int(np.floor(config.TEST_SIZE*len(df_returns)))
	train = df_returns.iloc[:-Ntest]
	test = df_returns.iloc[-Ntest:]


	return train, test

