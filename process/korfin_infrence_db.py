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
        company.append(i)        
    
company_df = pd.DataFrame(company, columns = ['company'])
text_df = pd.DataFrame(text, columns = ['text'])
result_df = pd.DataFrame(result, columns = ['result'])
all_df = pd.concat([company_df, text_df, result_df], axis=1)
all_df.to_csv(f'data/korfin_inference_db.csv', sep=',')
