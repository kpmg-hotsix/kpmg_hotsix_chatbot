import pandas as pd

keywordDB = pd.read_csv("data/keyword_DB.csv")
# name 은 list
# interface 쪽에서 3개 이상 차단.

def Tech_search(tech):
    keywords = []

    for i in reversed(range(len(tech)+1)):
        # print(i)
        for idx, keys in enumerate(keywordDB['keyword']):
            keys = eval(keys)
            matching = set(tech) & set(keys)
            if len(matching) == i:
                # print(i, tech, keys)
                keywords.append(keywordDB['company'][idx])
                if len(keywords) == 5:
                    break
    return keywords
    

# print(Tech_search('자율주행')[:5])