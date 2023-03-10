import json
import pandas as pd
import torch
import random
from datasets import Dataset
from functools import partial
import numpy as np
from transformers import AutoTokenizer

with open('../data/XXDX2100000001.json', encoding='utf-8-sig') as json_file: 
    json_data = json.load(json_file)

form = []
form_split = []
tmp = []
label = []
NE_form = []
begin = []
end = []
label_text=[]

for doc_idx in json_data['document']:
    for ne_idx in doc_idx['ne']:
        if ne_idx['label'] == 'TR_SCIENCE':
            for label_idx in ne_idx['examples']:  
                label_text.append('TR_SCIENCE')
                form.append(label_idx['form'])
                NE_form.append(label_idx['NE_form'])
                begin.append(label_idx['begin'])
                end.append(label_idx['end'])
                
        if ne_idx['label'] == 'TR_SOCIAL_SCIENCE':
            for label_idx in ne_idx['examples']:  
                label_text.append('TR_SOCIAL_SCIENCE')
                form.append(label_idx['form'])
                NE_form.append(label_idx['NE_form'])
                begin.append(label_idx['begin'])
                end.append(label_idx['end'])
                
        if ne_idx['label'] == 'TR_MEDICINE':
            for label_idx in ne_idx['examples']:  
                label_text.append('TR_MEDICINE')
                form.append(label_idx['form'])
                NE_form.append(label_idx['NE_form'])
                begin.append(label_idx['begin'])
                end.append(label_idx['end'])
                
        if ne_idx['label'] == 'TR_ART':
            for label_idx in ne_idx['examples']:  
                label_text.append('TR_ART')
                form.append(label_idx['form'])
                NE_form.append(label_idx['NE_form'])
                begin.append(label_idx['begin'])
                end.append(label_idx['end'])
                
        if ne_idx['label'] == 'TR_HUMANITIES':
            for label_idx in ne_idx['examples']:  
                label_text.append('TR_HUMANITIES')
                form.append(label_idx['form'])
                NE_form.append(label_idx['NE_form'])
                begin.append(label_idx['begin'])
                end.append(label_idx['end'])
                
        if ne_idx['label'] == 'TR_OTHERS':
            for label_idx in ne_idx['examples']:  
                label_text.append('TR_OTHERS')
                form.append(label_idx['form'])
                NE_form.append(label_idx['NE_form'])
                begin.append(label_idx['begin'])
                end.append(label_idx['end'])               
                
# print(len(label_text))
# print(len(form))
# print(len(NE_form))
# print(len(begin))
# print(len(end))
# print(label_text)
             
                
for i in form:
    form_split.append(list(i))
    
for j in form_split:    
    tmp.append([0 for i in range(len(j))])

for lt, l, b, e in zip(label_text, tmp, begin, end):
    if lt =='TR_SCIENCE':
        l[b:e] = [1 for i in range(b,e)]
        label.append(l)
        
    if lt =='TR_SOCIAL_SCIENCE':
        l[b:e] = [2 for i in range(b,e)]
        label.append(l)
        
    if lt =='TR_MEDICINE':
        l[b:e] = [3 for i in range(b,e)]
        label.append(l)
        
    if lt =='TR_ART':
        l[b:e] = [4 for i in range(b,e)]
        label.append(l)
        
    if lt =='TR_HUMANITIES':
        l[b:e] = [5 for i in range(b,e)]
        label.append(l)
        
    if lt =='TR_OTHERS':
        l[b:e] = [6 for i in range(b,e)]
        label.append(l)
        
all_df = pd.DataFrame({'data':form_split, 'label':label})

train_data = all_df[:int(len(all_df)*0.8)]
eval_data = all_df[int(len(all_df)*0.8):]

train_data.to_csv('../data/train_tr_data.csv', sep='\t')
eval_data.to_csv('../data/eval_tr_data.csv', sep='\t')

train_ds = Dataset.from_pandas(train_data, split="train")
eval_ds = Dataset.from_pandas(eval_data, split="eval")

class Loader :

    def __init__(self, max_length, tokenizer_name) :
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name


    def train_load(self) :
        global train_ds
        
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        encode_fn = partial(self.label_tokens_ner, tokenizer=tokenizer)

        train_ds = train_ds.map(encode_fn, batched=False)
        train_ds = train_ds.remove_columns(column_names=["data","label"])
        
        return train_ds
    
    def eval_load(self) :
        global eval_ds

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        encode_fn = partial(self.label_tokens_ner, tokenizer=tokenizer)

        eval_ds = eval_ds.map(encode_fn, batched=False)
        eval_ds = eval_ds.remove_columns(column_names=["data","label"])
        
        return eval_ds
    
    def label_tokens_ner(self, examples, tokenizer):
        sentence = "".join(examples["data"])
        tokenized_output = tokenizer(
            sentence,
            return_token_type_ids=False,
            return_offsets_mapping=True,
            max_length=self.max_length,
            truncation=True,
        )

        label_token_map = []

        list_label = examples["label"]
        list_label = [-100] + list_label + [-100]
        for token_idx, offset_map in enumerate(tokenized_output["offset_mapping"]):
            begin_letter_idx, end_letter_idx = offset_map
            label_begin = list_label[begin_letter_idx]
            label_end = list_label[end_letter_idx]
            
            token_label = np.array([label_begin, label_end])
            if label_begin == 6 and label_end == 6:
                token_label = 6
            elif label_begin == -100 and label_end == -100:
                token_label = -100
            else:
                token_label = label_begin if label_begin != 6 else 6
                token_label = label_end if label_end != 6 else 6

            label_token_map.append(token_label)

        tokenized_output["labels"] = label_token_map
        return tokenized_output
    
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
