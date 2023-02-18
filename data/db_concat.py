import pandas as pd
from itertools import chain

data = pd.read_csv("../data/HotsixDB.csv", sep = ',')

data = data.fillna('AI')
new_df = data.groupby('company').agg({'n_keyword': lambda x: list(x),
                                   'b_keyword': lambda x: list(x),
                                   'p_keyword': lambda x: list(x)}).reset_index()


new_df['keyword'] = new_df.apply(lambda x: list(set(x['n_keyword'] + x['b_keyword'] + x['p_keyword'])), axis=1)
new_df.drop(['n_keyword', 'b_keyword', 'p_keyword'], axis=1, inplace=True)


for i in range(0, len(new_df['keyword'])):
    new_df['keyword'][i] = list(set([item.strip() for sublist in [item.split(',') for item in new_df['keyword'][i]] for item in sublist]))

new_df.to_csv('../data/keyword_DB.csv', sep=',')
