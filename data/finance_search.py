import pandas as pd

def find_finance(name):
    finance = pd.read_csv('./data/finance.csv')
    idx = finance[finance['company'] == name].index
    return finance.iloc[idx]