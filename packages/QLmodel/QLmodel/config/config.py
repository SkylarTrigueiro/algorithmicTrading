import pathlib
import os
import datetime as dt
import pandas as pd
import QLmodel

pd.options.display.max_rows=20
pd.options.display.max_columns = 20

PACKAGE_ROOT = pathlib.Path(QLmodel.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'data'

# Data
START_DATE = start_date=dt.datetime.strptime("1997-01-01", "%Y-%m-%d")
END_DATE = end_date=dt.datetime.now()
TEST_SIZE = 0.2
STOCKS = ['MSFT', 'AAPL', 'JPM', 'PG']
TARGET = ['SPY']

# model
MODEL_NAME = 'model'
PIPELINE_NAME = 'pipe'

# Reinforcement Learning Parameters
NUM_EPISODES = 500

with open(os.path.join(PACKAGE_ROOT, 'VERSION')) as version_file:
    _version = version_file.read().strip()
    
MODEL_FILE_NAME = f'{MODEL_NAME}_{_version}.h5'
#MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, MODEL_FILE_NAME)

PIPELINE_FILE_NAME = f'{PIPELINE_NAME}_{_version}.pkl'
#PIPELINE_PATH = os.path.join(TRAINED_MODEL_DIR, PIPELINE_FILE_NAME)