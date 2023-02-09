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
def makeUrl(search, pg, op=0):
    start_page = makePgNum(pg)
    url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=" + search + "&start=" + str(start_page) + "&sort=" + str(op)
    print("생성url: ", url)
    return url

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
def getNaverURL(name, pg, op=0):

    url = makeUrl(name,pg, op)
    news_url = articles_crawler(url)

    # only Naver news
    final_urls = []
    for i in tqdm(range(len(news_url))):
        if "news.naver.com" in news_url[i]:
            final_urls.append(news_url[i])
        else:
            pass
    return final_urls

def getNewsTitle(urls, search):
    
    news_urls = []
    news_titles = []
    news_dates = []

    for i in urls:
        news = requests.get(i,headers=headers)
        news_html = BeautifulSoup(news.text,"html.parser")

        title = news_html.select_one("#ct > div.media_end_head.go_trans > div.media_end_head_title > h2")
        if title == None:
            title = news_html.select_one("#content > div.end_ct > div > h2")
        
        # # 제목에 기업명이 있는 경우만 추출
        # if search not in title:
        #     continue

        pattern1 = '<[^>]*>'
        title = re.sub(pattern=pattern1, repl='', string=str(title))

        news_urls.append(i)
        news_titles.append(title)

        try:
            html_date = news_html.select_one("div#ct> div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp > div > span") # 수정
            # html_date = news_html.find('div', {"class":"media_end_head_info nv_notrans "}).select_one("div.media_end_head_info_datestamp > div > span")
            news_date = html_date.attrs['data-date-time']
        except AttributeError:
            news_date = news_html.select_one("#content > div.end_ct > div > div.article_info > span > em")
            news_date = re.sub(pattern=pattern1,repl='',string=str(news_date))

        news_dates.append(news_date)
    return news_urls, news_titles, news_dates


def recent_check(dates):
    for date in dates:
        day = date.split()[0]
        now = datetime.now()
        date_to_compare = datetime.strptime(day, "%Y-%m-%d")
        date_diff = now - date_to_compare

        # 1 년 이상 차이 날 경우 false
        if date_diff.days > 365:
            return False
    return True


def news_search(name):
    
    ## set names
    name = "\""+ name +"\"" # for correct search
    news_urls = []
    news_titles = []
    news_dates = []

    for i in range(10):
        urls = getNaverURL(name, i+1, 1)
        news_urls2, news_titles2, news_dates2 = getNewsTitle(urls, name)
        news_urls += news_urls2
        news_titles += news_titles2
        news_dates += news_dates2
        if not recent_check(news_dates2):
            break
        if len(news_urls) >= 5:
            break

    return news_titles[:5]

