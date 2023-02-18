# -*- coding: utf-8 -*-

import ast, os, platform
from levenshtein_finder import levenshtein
import pandas as pd


root_dir = os.path.abspath(os.curdir)

_ = '\\' if platform.system() == 'Windows' else '/'
if root_dir[len(root_dir) - 1] != _: root_dir += _

BASE = {
    'root_dir': root_dir.format(_=_),
    'delimeter': _,  # OS에 따른 폴더 delimeter
}


def api(cls):
    for key, val in BASE.items():
        setattr(cls, key, val)
    return cls


def Typo_correction(text):
    data = pd.read_csv("dataset/finance.csv", sep=',')
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
    finance = pd.read_csv('./dataset/finance.csv')
    idx = finance[finance['company'] == name].index
    return finance.iloc[idx]

def find_sentiment(name):
    data = pd.read_csv('./dataset/korfin_inference_db.csv')
    idx = data[data['company'] == name].index
    return data.iloc[idx][['text','result']]

def similar_companies(name):
    similar = pd.read_csv('./dataset/similars.csv')
    similar['similars'] = similar['similars'].map(ast.literal_eval)
    idx = similar[similar['company'] == name].index
    sims = similar['similars'][idx].values.tolist()
    return sims


# name 은 list
# interface 쪽에서 3개 이상 차단.

def tech_search(tech):
    keywords = []
    keywordDB = pd.read_csv("dataset/keyword_DB.csv")

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