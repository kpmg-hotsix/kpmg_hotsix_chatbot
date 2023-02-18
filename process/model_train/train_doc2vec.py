import pandas as pd
import ast


df = pd.read_csv('../../data/keyword_DB.csv')
df['keyword'] = df['keyword'].map(ast.literal_eval)
# df = df[df['precise_intros'].map(lambda d: len(d)) > 0]

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [id]) for id, doc in zip(df['company'], df['keyword'])]
model = Doc2Vec(documents, vector_size=300, window=5, min_count=1, workers=4)

names = df['company'].unique().tolist()

similar = pd.DataFrame()

sims = []
for name in names:
    sims.append(model.dv.most_similar(name))
sims = [list(list(zip(*x))[0]) for x in sims]

similar['name'] = names
similar['similars'] = sims
similar.to_csv('../../data/similars.csv', index=False)