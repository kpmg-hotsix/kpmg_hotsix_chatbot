import pandas as pd
import ast

def train():
    df = pd.read_csv('../../dataset/kpmg_news_db.csv')
    df['precise_intros'] = df['precise_intros'].map(ast.literal_eval)
    df = df[df['precise_intros'].map(lambda d: len(d)) > 0]

    from pecab import PeCab

    user_dict = df['company'].unique().tolist()
    pecab = PeCab(user_dict=user_dict)

    df['precise_intros'] = df['precise_intros'].map(lambda x: ' '.join(x))
    df['token'] = df['precise_intros'].map(pecab.morphs)

    from gensim.models.doc2vec import Doc2Vec, TaggedDocument

    documents = [TaggedDocument(doc, [id]) for id, doc in zip(df['search'], df['token'])]
    model = Doc2Vec(documents, vector_size=300, window=5, min_count=1, workers=4)

    names = df['search'].unique().tolist()

    similar = pd.DataFrame()

    sims = []
    for name in names:
        sims.append(model.dv.most_similar(name))
    sims = [list(list(zip(*x))[0]) for x in sims]

    similar['name'] = names
    similar['similars'] = sims