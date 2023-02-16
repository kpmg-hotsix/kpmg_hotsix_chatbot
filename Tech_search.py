import pandas as pd

data = pd.read_csv("data/keyword_DB.csv", sep = ',')
tmp = []

def Tech_search(name):
    for idx, key in enumerate(data['keyword']):
        if name in key:
            tmp.append(data['company'][idx])
    return tmp
    

print(Tech_search('자율주행')[:5])