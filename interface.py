import re
from data.news_search import news_search
from Sentiment_Analysis.korfin_inference import Sentiment_Analysis

while True:
    t = input("\nchat: ")
    '''
    task1_text = "자연어 처리 기술과 관련된 기업 알려줘"
    task2_text = "콴다 기업과 유사한 기업 찾아줘"
    task3_text = "콴다 기업의 최근 이슈 알려줘"
    task4_text = "콴다 기업의 재무재표 알려줘"
    '''

    if "기술과" in t:
        print(t.split("기술과")[0])
        print('task1') 

    if "유사한 기업" in t:
        company = t.split("유사한 기업")[0].replace('과', '')
        print(company) 
        print('task2') 

    if "최근 이슈" in t:
        company = t.split("최근 이슈")[0].replace('의', '')
        print(company) 
        print('task3') 
        news_title = news_search(company)
        print(news_title)
        for t in news_title:
            Sentiment_Analysis(t)
        
    if "재무재표" in t:
        company = t.split("재무재표")[0].replace('의', '')
        print(company) 
        print('task4') 