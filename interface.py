import re
from data.news_search import news_search
from Sentiment_Analysis.korfin_inference import Sentiment_Analysis
from data.company_search import similar_companies
from data.finance_search import find_finance
from Typo_correction import Typo_correction

while True:
    t = input("\nchat: ")
    '''
    example:
    
    task1_text = "자연어 처리 기술과 관련된 기업 알려줘"
    task2_text = "매스프레소와(과) 유사한 기업 찾아줘"
    task3_text = "매스프레소의 최근 이슈 알려줘"
    task4_text = "매스프레소의 재무제표 알려줘"
    '''

    if "기술과" in t:
        print(t.split("기술과")[0])
        print('task1') 

    if "유사한 기업" in t:
        # company = t.split("유사한 기업")[0].replace('과', '')
        company = re.split('[과|와]', t)[0] #'과' 말고도 '와'도 경우의 수에 들어가므로..
        typo_company = Typo_correction(company)
        # print('task2')
        similar_company = similar_companies(typo_company)
        print('\n'.join(similar_company[0]))


    if "최근 이슈" in t:
        # company = t.split("최근 이슈")[0].replace('의', '')
        company = re.split('의', t)[0]
        # print(company) 
        # print('task3') 
        typo_company = Typo_correction(company)
        news_title = news_search(typo_company)
        for t in news_title:
            title, result = Sentiment_Analysis(t)
            print(f"title: {title}", f"Result: {result}")
                
    if "재무제표" in t:
        # company = t.split("재무제표")[0].replace('의', '')
        company = re.split('의', t)[0]
        typo_company = Typo_correction(company)
        # print(typo_company)
        # print('task4')
        fin = find_finance(typo_company)
        # print(fin)
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
