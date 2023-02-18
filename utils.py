# -*- coding: utf-8 -*-

import ast
from levenshtein_finder import levenshtein
import pandas as pd


def typo_correction(text):
    data = pd.read_csv("./data/finance.csv", sep=',')
    data = data['company']
    distance = {word:levenshtein(word, text) for word in data}
    similars = sorted(filter(lambda x:x[1] <= 1, distance.items()), key=lambda x:x[1])
    matching = []
    # matching = [s for s in data if similars[0][0] in s]
    for s in data:
        if similars[0][0] in s:
            matching.append(s)
        else:
            pass
        
    return matching[0]

def find_finance(name):
    finance = pd.read_csv('./data/finance.csv')
    finance = finance.query(f"company == '{name}'")
    return finance

def find_sentiment(name):
    data = pd.read_csv('./data/korfin_inference_db.csv')
    data = data.query(f"company == '{name}'")
    return data

def similar_companies(name):
    similar = pd.read_csv('./data/similars.csv')
    similar['similars'] = similar['similars'].map(ast.literal_eval)
    similar = similar.query(f"company == '{name}'")
    sims = similar['similars'].values.tolist()
    return sims


# name 은 list
# interface 쪽에서 3개 이상 차단.

def tech_search(tech):
    keywords = []
    keywordDB = pd.read_csv("data/keyword_DB.csv")

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