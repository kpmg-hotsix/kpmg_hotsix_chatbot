import re
from data.news_search import news_search
from Sentiment_Analysis.korfin_inference import Sentiment_Analysis

from data.company_search import similar_companies
from data.finance_search import find_finance

while True:
    t = input("\nchat: ")
    '''
    example:
    
    task1_text = "자연어 처리 기술과 관련된 기업 알려줘"
    task2_text = "콴다와(과) 유사한 기업 찾아줘"
    task3_text = "콴다 기업의 최근 이슈 알려줘"
    task4_text = "콴다의 재무재표 알려줘"
    '''

    if "기술과" in t:
        print(t.split("기술과")[0])
        print('task1') 

    if "유사한 기업" in t:
        # company = t.split("유사한 기업")[0].replace('과', '')
        company = re.split('[과|와]', t)[0] #'과' 말고도 '와'도 경우의 수에 들어가므로..
        print(company) 
        print('task2')
        similar_company = similar_companies(company)
        print('\n'.join(similar_company[0]))


    if "최근 이슈" in t:
        company = t.split("최근 이슈")[0].replace('의', '')
        print(company) 
        print('task3') 
        news_title = news_search(company)
        print(news_title)
        for t in news_title:
            Sentiment_Analysis(t)
        
    if "재무제표" in t:
        # company = t.split("재무제표")[0].replace('의', '')
        company = re.split('의', t)[0]
        print(company)
        print('task4')
        fin = find_finance(company)
        print(f'''
        유동비율: {fin['유동비율'].values[0]}
        자기자본비율: {fin['자기자본비율'].values[0]}
        부채비율: {fin['부채비율'].values[0]}
        총자산회전율: {fin['총자산회전율'].values[0]}
        총자산증가율: {fin['총자산증가율'].values[0]}
        매출액증가율: {fin['매출액증가율'].values[0]}
        순이익증가율: {fin['순이익증가율'].values[0]}

        (기준년도: {fin['기준년도'].values[0]})
        ''')
