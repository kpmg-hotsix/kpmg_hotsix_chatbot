from bs4 import BeautifulSoup
import requests
import re
import datetime
from tqdm import tqdm
import sys
import pandas as pd


# ConnectionError 방지
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"}


def makePgNum(num):
    if num == 1:
        return num
    elif num == 0:
        return num+1
    else:
        return num+9*(num-1)


# 크롤링 url 생성 (검색어, 시작, 종료)
# &sort= 0 관련순 1 최신순 2 오래된순
def makeUrl(search, start_pg, end_pg):
    if start_pg == end_pg:
        start_page = makePgNum(start_pg)
        url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=" + search + "&start=" + str(start_page)
        print("생성url: ", url)
        return [url]
    else:
        urls = []
        for i in range(start_pg, end_pg + 1):
            page = makePgNum(i)
            url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=" + search + "&start=" + str(page)
            urls.append(url)
        print("생성url: ", urls)
        return urls    

def news_attrs_crawler(articles,attrs):
    attrs_content=[]
    for i in articles:
        attrs_content.append(i.attrs[attrs])
    return attrs_content

# get urls
def articles_crawler(url):
    # html parsing
    original_html = requests.get(url,headers=headers)
    html = BeautifulSoup(original_html.text, "html.parser")

    url_naver = html.select("div.group_news > ul.list_news > li div.news_area > div.news_info > div.info_group > a.info")
    url = news_attrs_crawler(url_naver,'href')
    return url


# naver url 생성
def getNaverURL(name, pg):
    news_url =[]

    url = makeUrl(name,pg,pg)

    for i in url:
        url = articles_crawler(i)
        news_url.append(url)

    # 1차원 리스트로
    news_url_1 = []
    news_url_1 = sum(news_url, [])

    # NAVER 뉴스만 남기기
    final_urls = []
    for i in tqdm(range(len(news_url_1))):
        if "news.naver.com" in news_url_1[i]:
            final_urls.append(news_url_1[i])
        else:
            pass
    return final_urls

# 뉴스 내용 크롤링
def getNewsContents(urls, name):
    
    news_titles = []
    news_contents =[]
    news_dates = []
    news_urls = []

    for i in urls:
        # 기사 html get
        news = requests.get(i,headers=headers)
        news_html = BeautifulSoup(news.text,"html.parser")

        # 뉴스 제목 
        title = news_html.select_one("#ct > div.media_end_head.go_trans > div.media_end_head_title > h2")
        
        if title == None:
            title = news_html.select_one("#content > div.end_ct > div > h2")
            if title == None :
                continue
        
        content = news_html.select("div#dic_area")
        if content == []: # content 추출 안 됐을 때
            content = news_html.select("#articeBody")

        print(i, len(content))
        content = ''.join(str(content))

        pattern1 = '<[^>]*>'
        title = re.sub(pattern=pattern1, repl='', string=str(title))
        if name not in title:
            continue

        content = re.sub(pattern=pattern1, repl='', string=content)
        pattern2 = """[\n\n\n\n\n// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback() {}"""
        content = content.replace(pattern2, '')
        
        news_urls.append(i)
        news_titles.append(title)
        news_contents.append(content)

        try:
            html_date = news_html.select_one("div#ct> div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp > div > span") # 수정
            # html_date = news_html.find('div', {"class":"media_end_head_info nv_notrans "}).select_one("div.media_end_head_info_datestamp > div > span")
            news_date = html_date.attrs['data-date-time']
        except AttributeError:
            news_date = news_html.select_one("#content > div.end_ct > div > div.article_info > span > em")
            news_date = re.sub(pattern=pattern1,repl='',string=str(news_date))
        # 날짜 가져오기
        news_dates.append(news_date)
    return news_urls, news_titles, news_contents, news_dates



final_news_titles = []
final_news_urls = []
final_news_contents = []
final_news_dates = []
final_news_searches = []

# set names
raw_data = pd.read_csv('./data/company_list.csv', encoding='utf-8-sig')
print(raw_data.head())
names = raw_data['company'].to_list()
names = ["\""+i+"\"" for i in names] # for correct search
# names = ['\"노타\"']

for name in tqdm(names):
    news_urls = []
    news_titles = []
    news_dates = []
    news_contents = []
    for cnt in range(10):
        urls = getNaverURL(name, cnt+1)
        news_urls2, news_titles2, news_contents2, news_dates2 = getNewsContents(urls, name[1:-1])
        news_urls += news_urls2
        news_titles += news_titles2
        news_contents += news_contents2
        news_dates += news_dates2
        if len(news_urls) >= 5:
            break
    print('news_title: ',len(news_titles))
    print('news_url: ',len(news_urls))
    print('news_contents: ',len(news_contents))
    print('news_dates: ',len(news_dates))
    final_news_searches += [name[1:-1]]*len(news_titles[:5])
    final_news_titles += news_titles[:5]
    final_news_urls += news_urls[:5]
    final_news_contents += news_contents[:5]
    final_news_dates += news_dates[:5]

print(len(final_news_searches), len(final_news_titles), len(final_news_contents), len(final_news_urls), len(final_news_dates))

### dataframe save
news_df = pd.DataFrame({'search': final_news_searches,'date':final_news_dates,'title':final_news_titles,'link':final_news_urls,'content':final_news_contents})
news_df = news_df.drop_duplicates(keep='first',ignore_index=True)
print("중복 제거 후 행 개수: ",len(news_df))

now = datetime.datetime.now() 
news_df.to_csv('{}.csv'.format(now.strftime('%Y%m%d_%Hh%Mm%Ss')),encoding='utf-8-sig',index=False)