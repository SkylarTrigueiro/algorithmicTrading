import pandas as pd
import quandl
import datetime as dt
import os


def create_data():
	symbols_table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
								 header=0)[0]
	symbols = list(symbols_table.loc[:, "Symbol"])

	quandl.ApiConfig.api_key = 'NzxBaUswV2awx4VudjHU'
	for symbol in symbols:
		if not os.path.exists(f"data/{symbol}.csv"):
			try:
				data = quandl.get("WIKI/" + symbol, start_date=dt.datetime.strptime("1997-01-01", "%Y-%m-%d"), end_date=dt.datetime.now())
				if data.size > 0:
					data.to_csv(f"data/{symbol}.csv")
					print(f"{symbol} data saved.")
			except:
				print('not saving')

def load_data(symbols=None):

	dates = pd.date_range(start = '2006-01-01', end = '2018-01-01', freq='1D')
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

		temp = pd.read_csv(f'data/{symbol}.csv')
		temp = temp[['Date','Adj. Close']]
		temp.columns = ['Date', str(symbol)]
		temp['Date'] = pd.to_datetime(temp['Date'], format='%Y-%m-%d')
		temp = temp.set_index('Date')
		df = df.join(temp, how='left' )

	return df

