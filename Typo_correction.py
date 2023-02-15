# -*- coding: utf-8 -*-

import time
from levenshtein_finder import levenshtein
import pandas as pd

data = pd.read_csv("data/final_finance_company.csv", sep=',')
data = data['company']

def Typo_correction(text):
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