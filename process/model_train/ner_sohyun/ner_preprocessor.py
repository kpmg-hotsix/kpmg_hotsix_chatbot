import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def read_json(file_name):
    form = []
    ne = []

    with open(file_name) as f:
        json_data = json.loads(f.read())

    for document in json_data['document']:
        for sentence in document['sentence']:
            if len(sentence['NE']) > 0:
                form.append(sentence['form'])
                ne.append(sentence['NE'])
    df = pd.DataFrame(dict(data=form, label=ne)) 
    return df
    
def arrange_data():
    data = list()
    data_path = '/Users/simso/Project/ner/data/json/'
    path1 = os.listdir(data_path + '21_150tags_NamedEntity/SXNE21')
    file_list = [data_path+'19_150tags_NamedEntity/NXNE2102008030.json'] + [data_path+'21_150tags_NamedEntity/SXNE21/'+json_file for json_file in path1]
    for file_name in file_list:
        temp_data = read_json(file_name)
        data.append(temp_data)
    all_data = pd.concat(data)
    # train_data, eval_data = train_test_split(all_data, test_size=0.2, random_state=random_state)
    # train_data.to_csv("data/train_data.csv", sep="\t", index=False)
    # eval_data.to_csv("data/eval_data.csv", sep="\t", index=False)
    all_data.to_csv("../data/ner_data.csv", sep='\t', index=False)

arrange_data()