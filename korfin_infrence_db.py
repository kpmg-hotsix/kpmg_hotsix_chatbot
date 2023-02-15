import pandas as pd
from data.news_search import news_search
from Sentiment_Analysis.korfin_inference import Sentiment_Analysis


data = pd.read_csv("data/final_finance_company.csv")

text = []
result = []
company = []

for i in data['company']:
    news_title = news_search(i)
    for t in news_title:
        t, r = Sentiment_Analysis(t)
        text.append(t)
        result.append(r)
                
    
text_df = pd.DataFrame(text, columns = ['text'])
result_df = pd.DataFrame(result, columns = ['result'])

        
label_idx = 0
for i in range(0, len(text_df), 5):
    text_df.loc[i:i+4, 'company'] = data['company'][label_idx]
    label_idx += 1
    if label_idx == len(data['company']):
        label_idx = 0

all_df = pd.concat([text_df, result_df], axis=1)
all_df.to_csv(f'result/korfin_inference_db.csv', sep=',')
