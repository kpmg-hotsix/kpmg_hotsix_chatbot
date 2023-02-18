import pandas as pd
import datetime
import re

def getPrecise(word):
    flag = True
    if "가" <= word[-1] <= "힣":
        flag = (ord(word[-1])-ord("가")) % 28 > 0
    if flag :
        return word + '은'
    else:
        return word + '는'


raw_data = pd.read_csv('final_news.csv')

contents = raw_data['content'].apply(lambda x : re.sub('\n{3,5}', '\n', x))
searches = raw_data['search']
intros = []
precise_intros = []

for i in range(len(contents)):
    intro = []
    precise_intro = []

    lines = contents[i].split('.')
    precise = getPrecise(searches[i])

    for line in lines:
        if precise in line:
            precise_intro.append(line)
        if searches[i] in line:
            intro.append(line)
    
    intros.append(intro)
    precise_intros.append(precise_intro)

intro_df = pd.DataFrame(columns=['search', 'title', 'link', 'content', 'precise_intros', 'intros', 'date'])

intro_df['search'] = searches
intro_df['title'] = raw_data['title']
intro_df['link'] = raw_data['link']
intro_df['content'] = contents
intro_df['precise_intros'] = precise_intros
intro_df['intros'] = intros
intro_df['date'] = raw_data['date']

now = datetime.datetime.now() 
intro_df.to_csv('{}.csv'.format(now.strftime('%Y%m%d_%Hh%Mm%Ss')),encoding='utf-8-sig',index=False)
# tmpdf.to_csv('testing.csv', encoding='utf-8-sig',index=False)
